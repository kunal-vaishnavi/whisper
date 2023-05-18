import argparse
import time
import torch
import whisper

def measure(args):
    model = whisper.load_model(args.size)
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, torch.qint8).to("cpu")
    qmodel = qmodel if args.no_compile else torch.compile(qmodel)
    options = whisper.DecodingOptions(beam_size=args.beam_size, fp16=not args.no_fp16)

    audio = None
    start_time = time.time()
    for _ in range(num_iter):
        audio = whisper.load_audio(args.audio)
        audio = whisper.pad_or_trim(audio)
    end_time = time.time()
    print(f"PyTorch load audio: {(end_time - start_time) / num_iter} s")

    def e2e(audio):
        mel = whisper.log_mel_spectrogram(audio).to(qmodel.device)
        result = whisper.decode(qmodel, mel, options)
        return result

    # Warm up
    for _ in range(args.warmup_runs):
        result = e2e(audio)

    start_time = time.time()
    for _ in range(ags.num_runs):
        e2e(audio)
    end_time = time.time()

    print(f"PyTorch E2E: {(end_time - start_time) / num_iter} s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("-s", "--size", type=str, required=True, help="Size of model")

    parser.add_argument("-b", "--beam-size", type=int, required=True, help="Beam size for beam search")
    parser.add_argument("-w", "--warmup-runs", type=int, default=5, required=False, help="Number of warmup runs before measuring")
    parser.add_argument("-r", "--num-runs", type=int, default=100, required=False, help="Number of inference runs to measure")

    parser.add_argument("--no-compile", action="store_true", help="Turn off torch.compile(model)")
    parser.set_defaults(no_compile=False)
    parser.add_argument("--no-fp16", action="store_true", help="Turn off FP16 ")
    parser.set_defaults(no_fp16=False)

if __name__ == "__main__":
    main()
