import asyncio
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from meshcore import EventType, MeshCore

from keyboard_ptt import KeyboardPttController
from stt import WhisperTranscriber
from tts import PiperTTS

PROJECT_NAME = "meshie-talkie"
TMP_DIR = Path("/tmp") / PROJECT_NAME
SERIAL_PORT = "/dev/ttyACM0"
MESH_CHANNEL = 2


class MeshBridge:
    def __init__(self, serial_port: str, channel: int, on_message=None):
        self.serial_port = serial_port
        self.channel = channel
        self.on_message = on_message
        self._meshcore = None
        self._loop = None
        self._send_queue = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, daemon=True, name="mesh-bridge")
        self._thread.start()
        self._ready_event.wait(timeout=10)
        return self._ready_event.is_set()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def send_text(self, text: str) -> bool:
        if not text or self._loop is None or self._send_queue is None:
            return False
        self._loop.call_soon_threadsafe(self._send_queue.put_nowait, text)
        return True

    def _run(self):
        asyncio.run(self._run_async())

    async def _run_async(self):
        try:
            self._meshcore = await MeshCore.create_serial(self.serial_port)
            print(f"\nMesh connected on {self.serial_port}")
        except Exception as exc:
            print(f"\nMesh connection failed: {exc}")
            self._ready_event.set()
            return

        self._loop = asyncio.get_running_loop()
        self._send_queue = asyncio.Queue()
        self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                try:
                    while True:
                        try:
                            text = self._send_queue.get_nowait()
                        except Exception:
                            break
                        result = await self._meshcore.commands.send_chan_msg(self.channel, text)
                        if result.type == EventType.ERROR:
                            print(f"\nMesh send failed: {result.payload}")

                    result = await self._meshcore.commands.get_msg(timeout=0.2)
                    if result.type == EventType.NO_MORE_MSGS:
                        await asyncio.sleep(0.05)
                    elif result.type == EventType.CHANNEL_MSG_RECV:
                        msg = result.payload
                        if msg.get("channel_idx") == self.channel:
                            text = msg.get("text", "").strip()
                            if text and self.on_message is not None:
                                self.on_message(text)
                    elif result.type == EventType.CONTACT_MSG_RECV:
                        msg = result.payload
                        text = msg.get("text", "").strip()
                        if text and self.on_message is not None:
                            self.on_message(text)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    print(f"\nMesh loop error: {exc}")
                    await asyncio.sleep(0.5)
        finally:
            if self._meshcore is not None:
                try:
                    await self._meshcore.disconnect()
                except Exception:
                    pass


def write_wav_file(filename: Path, sample_rate: int, samples: np.ndarray):
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def play_wav_file(path: Path):
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as exc:
        print(f"Audio playback error: {exc}")


def sendMC(text: str, mesh_bridge: MeshBridge):
    if mesh_bridge.send_text(text):
        print(f"Sent via mesh: {text}")
    else:
        print(f"Mesh unavailable; would send: {text}")


def main():
    print("Starting meshie talkie...")

    stt = WhisperTranscriber()
    tts = PiperTTS(output_dir=TMP_DIR)
    print(f"Using STT model dir: {stt.model_dir}")

    def handle_mesh_message(text: str):
        print(f"\nMesh message received: {text}")
        output_path = tts.synthesize(text)
        print(f"Saved TTS audio to {output_path}")
        play_wav_file(output_path)

    mesh_bridge = MeshBridge(
        serial_port=SERIAL_PORT,
        channel=MESH_CHANNEL,
        on_message=handle_mesh_message,
    )
    mesh_bridge.start()

    print("Started! Hold 't' to record, then release to transcribe with Whisper\n")
    print(f"Mesh channel {MESH_CHANNEL} is configured for outgoing and incoming text")

    sample_rate = 16000
    buffer_lock = threading.Lock()
    captured_frames = []
    transcribe_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        with buffer_lock:
            if ptt_controller.is_recording:
                captured_frames.append(indata.copy())

    def process_recording(samples: np.ndarray):
        temp_dir = TMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)

        wav_path = Path(temp_dir / "input_stt.wav")

        write_wav_file(wav_path, sample_rate, samples)
        try:
            print(f"\nSaved clip to {wav_path}")

            with transcribe_lock:
                result = stt.transcribe_file(wav_path)

            if result:
                #print(f"Transcript: {result}")
                sendMC(result, mesh_bridge)
            else:
                print("Transcript: <empty>")
        finally:
            print("\nReady for next recording. Hold 't' to record again.")

    def start_recording():
        with buffer_lock:
            captured_frames.clear()
        print("\nRecording... hold 't' until you finish speaking", end="", flush=True)

    def stop_recording():
        with buffer_lock:
            frames = [frame.copy() for frame in captured_frames]
            captured_frames.clear()

        if not frames:
            print("\nNo audio captured.")
            return

        samples = np.concatenate(frames, axis=0).reshape(-1)
        threading.Thread(target=process_recording, args=(samples,), daemon=True).start()

    ptt_controller = KeyboardPttController(
        hold_timeout=0.7,
        restart_cooldown=1.0,
        on_record_start=start_recording,
        on_record_stop=stop_recording,
    )

    try:
        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=sample_rate,
            callback=audio_callback,
        ):
            ptt_controller.run()
    finally:
        mesh_bridge.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
