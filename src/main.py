import sys
import tempfile
import threading
import select
import termios
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

project_name = "meshie-talkie"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "sherpa-onnx-whisper-tiny.en"


def ensure_file_exists(path: Path):
    assert path.is_file(), (
        f"{path} does not exist!\n"
        "Please check that the model files are in src/models."
    )


def write_wav_file(filename: Path, sample_rate: int, samples: np.ndarray):
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def transcribe_wav_file(recognizer, wav_path: Path) -> str:
    stream = recognizer.create_stream()

    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Recorded audio must be mono")
        sample_rate = wav_file.getframerate()

        while True:
            frames = wav_file.readframes(4096)
            if not frames:
                break
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(sample_rate, samples)

            if hasattr(recognizer, "is_ready"):
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

    if hasattr(stream, "input_finished"):
        stream.input_finished()

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
    else:
        recognizer.decode_stream(stream)

    if hasattr(recognizer, "get_result"):
        result = recognizer.get_result(stream)
        return getattr(result, "text", result).strip()

    return getattr(stream.result, "text", str(stream.result)).strip()


class TerminalRawInput:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.previous_settings = termios.tcgetattr(self.fd)
        self.raw_settings = termios.tcgetattr(self.fd)
        self.raw_settings[3] = self.raw_settings[3] & ~(termios.ECHO | termios.ICANON)
        self.raw_settings[6][termios.VMIN] = 1
        self.raw_settings[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.raw_settings)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous_settings)

def create_recognizer(model_dir: Path):
    for filename in ("tiny.en-encoder.int8.onnx", "tiny.en-decoder.int8.onnx", "tiny.en-tokens.txt"):
        ensure_file_exists(model_dir / filename)

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(model_dir / "tiny.en-encoder.int8.onnx"),
        decoder=str(model_dir / "tiny.en-decoder.int8.onnx"),
        tokens=str(model_dir / "tiny.en-tokens.txt"),
        num_threads=1,
        decoding_method="greedy_search",
        provider="cpu",
    )
    return recognizer

#*******************************main()*******************************#

def main():
    print("Starting STT")

    model_dir = DEFAULT_MODEL_DIR
    print(f"Using model_dir: {model_dir}")
    recognizer = create_recognizer(model_dir)
    print("Started! Hold 't' to record, then release to transcribe with Whisper")

    sample_rate = 16000
    hold_timeout = 0.7
    restart_cooldown = 1.0
    recording_active = threading.Event()
    buffer_lock = threading.Lock()
    captured_frames = []
    transcribe_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        with buffer_lock:
            if recording_active.is_set():
                captured_frames.append(indata.copy())

    def process_recording(samples: np.ndarray):
        temp_dir = Path("/tmp") / project_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir) as tmp_file:
            wav_path = Path(tmp_file.name)

        write_wav_file(wav_path, sample_rate, samples)
        try:
            print(f"\nSaved clip to {wav_path}")

            with transcribe_lock:
                result = transcribe_wav_file(recognizer, wav_path)

            print(f"Transcript: {result}" if result else "Transcript: <empty>")
        finally:
            wav_path.unlink(missing_ok=True) # Clean up the temporary file

    def start_recording():
        with buffer_lock:
            captured_frames.clear()

        recording_active.set()
        print("\nRecording... hold 't' until you finish speaking", end="", flush=True)

    def stop_recording():
        if not recording_active.is_set():
            return

        recording_active.clear()

        with buffer_lock:
            frames = [frame.copy() for frame in captured_frames]
            captured_frames.clear()

        if not frames:
            print("\nNo audio captured.")
            return

        samples = np.concatenate(frames, axis=0).reshape(-1)
        threading.Thread(target=process_recording, args=(samples,), daemon=True).start()

    def poll_keyboard_input():
        last_t_time = None
        cooldown_until = 0.0
        with TerminalRawInput():
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    char = sys.stdin.read(1)
                    if char == "\x03":
                        raise KeyboardInterrupt
                    if char.lower() == "t":
                        now = time.monotonic()
                        if now < cooldown_until:
                            continue

                        if not recording_active.is_set():
                            start_recording()
                        last_t_time = now

                if recording_active.is_set() and last_t_time is not None:
                    now = time.monotonic()
                    if now - last_t_time >= hold_timeout:
                        stop_recording()
                        cooldown_until = now + restart_cooldown
                        last_t_time = None

    with sd.InputStream(
        channels=1,
        dtype="float32",
        samplerate=sample_rate,
        callback=audio_callback,
    ):
        poll_keyboard_input()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)