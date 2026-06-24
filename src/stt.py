import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_STT_MODEL_DIR = DEFAULT_MODELS_DIR / "sherpa-onnx-whisper-tiny.en"


class WhisperTranscriber:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir or DEFAULT_STT_MODEL_DIR)
        self._ensure_model_files()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(self.model_dir / "tiny.en-encoder.int8.onnx"),
            decoder=str(self.model_dir / "tiny.en-decoder.int8.onnx"),
            tokens=str(self.model_dir / "tiny.en-tokens.txt"),
            language="en",
            task="transcribe",
            num_threads=1,
            decoding_method="greedy_search",
            provider="cpu",
        )

    def _ensure_model_files(self):
        required = (
            "tiny.en-encoder.int8.onnx",
            "tiny.en-decoder.int8.onnx",
            "tiny.en-tokens.txt",
        )
        for filename in required:
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing speech model file: {path}")

    def transcribe_file(self, wav_path: str | Path) -> str:
        wav_path = Path(wav_path)
        stream = self.recognizer.create_stream()

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

                if hasattr(self.recognizer, "is_ready"):
                    while self.recognizer.is_ready(stream):
                        self.recognizer.decode_stream(stream)

        if hasattr(stream, "input_finished"):
            stream.input_finished()
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
        else:
            self.recognizer.decode_stream(stream)

        if hasattr(self.recognizer, "get_result"):
            result = self.recognizer.get_result(stream)
            return getattr(result, "text", result).strip()

        return getattr(stream.result, "text", str(stream.result)).strip()
