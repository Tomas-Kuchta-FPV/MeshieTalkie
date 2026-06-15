# tts.py should take in a text string and return the path to a wav file containing the generated speech. It uses lib sherpa-onnx.
import sherpa_onnx
import soundfile as sf
import main

DEFAULT_TTS_MODEL_DIR = main.DEFAULT_MODELS_DIR / "vits-piper-en_US-amy-low"

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{DEFAULT_TTS_MODEL_DIR}/en_US-amy-low.onnx",
            data_dir=f"{DEFAULT_TTS_MODEL_DIR}/espeak-ng-data",
            tokens=f"{DEFAULT_TTS_MODEL_DIR}/tokens.txt",
        ),
        num_threads=1,
    ),
)

if not config.validate():
    raise ValueError(f"Please check your path {config.model.vits.model} and ensure the \"vits-piper-en_US-amy-low\" model is installed")

def tts_generate(text: str, sid: int = 0, speed: float = 1.0):
    tts = sherpa_onnx.OfflineTts(config)
    audio = tts.generate(text=text, sid=sid, speed=speed)
    audio_path = sf.write(f"{main.TMP_DIR}/output_tts.wav", audio.samples, samplerate=audio.sample_rate)
    return audio_path

