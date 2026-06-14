import sys
from pathlib import Path

import sounddevice as sd
import sherpa_onnx
from types import SimpleNamespace



BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please check that the model files are in src/models."
    )

def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
    )
    return recognizer

#*******************************main()*******************************#

def main():
    print("Starting STT")

    model_dir = Path(str(DEFAULT_MODEL_DIR))
    print(f"Using model_dir: {model_dir}")
    args = SimpleNamespace(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(model_dir / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(model_dir / "joiner-epoch-99-avg-1.int8.onnx"),
        decoding_method="greedy_search",
        max_active_paths=4,
        provider="cpu",
        hotwords_file="",
        hotwords_score=1.5,
        blank_penalty=0.0,
        hr_rule_fsts="",
        hr_lexicon="",
    )

    recognizer = create_recognizer(args)
    print("Started! Please speak")

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)
    last_result = ""
    stream = recognizer.create_stream()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            if last_result != result:
                last_result = result
                print("\r{}".format(result), end="", flush=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)