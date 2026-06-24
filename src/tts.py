from pathlib import Path

import sherpa_onnx
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_TTS_MODEL_DIR = DEFAULT_MODELS_DIR / "vits-piper-en_US-amy-low"
DEFAULT_OUTPUT_DIR = Path("/tmp") / "meshie-talkie"


class PiperTTS:
    def __init__(self, model_dir: Path | None = None, output_dir: Path | None = None):
        self.model_dir = Path(model_dir or DEFAULT_TTS_MODEL_DIR)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(self.model_dir / "en_US-amy-low.onnx"),
                    data_dir=str(self.model_dir / "espeak-ng-data"),
                    tokens=str(self.model_dir / "tokens.txt"),
                ),
                num_threads=1,
            ),
        )

        if not self.config.validate():
            raise ValueError(
                f"Please check your path {self.config.model.vits.model} and ensure the \"vits-piper-en_US-amy-low\" model is installed"
            )

    def synthesize(self, text: str, sid: int = 0, speed: float = 1.0, output_path: Path | str | None = None):
        if output_path is None:
            output_path = self.output_dir / "output_tts.wav"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tts = sherpa_onnx.OfflineTts(self.config)
        audio = tts.generate(text=text, sid=sid, speed=speed)
        sf.write(str(output_path), audio.samples, samplerate=audio.sample_rate)
        return output_path

