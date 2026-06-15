# stt.py should take in a path of a wav file and return the transcript as text. It uses lib sherpa-onnx.
import sherpa_onnx

def stt_transcribe(wav_path: str) -> str:
    config = sherpa_onnx.OfflineRecognizerConfig(
        model=sherpa_onnx.OfflineRecognizerModelConfig(
            encoder=sherpa_onnx.OfflineRecognizerEncoderModelConfig(
                model="models/sherpa-onnx-whisper-tiny.en/encoder-quant.onnx",
                tokens="models/sherpa-onnx-whisper-tiny.en/tokens.txt",
            ),
            decoder=sherpa_onnx.OfflineRecognizerDecoderModelConfig(
                model="models/sherpa-onnx-whisper-tiny.en/decoder-quant.onnx",
            ),
            joiner=sherpa_onnx.OfflineRecognizerJoinerModelConfig(
                model="models/sherpa-onnx-whisper-tiny.en/joiner-quant.onnx",
            ),
        ),
        num_threads=1,
    )

    if not config.validate():
        raise ValueError("Please install the \"sherpa-onnx-whisper-tiny.en\" model")

    recognizer = sherpa_onnx.OfflineRecognizer(config)
    result = recognizer.transcribe(wav_path)
    return result.text