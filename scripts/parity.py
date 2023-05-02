########################################################
# Verify Whisper model
#
# See benchmark.sh for examples on how to run this file
########################################################

import argparse
import librosa
import numpy as np
import onnxruntime
import time
import torch
import torch.autograd.profiler as profiler
import whisper

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

PRECISION = {
    "fp32": (torch.float32, np.float32),
    "fp16": (torch.float16, np.float16),
    "int8": (torch.int8, np.int8),
}

def get_ort_inputs(args, processor):
    if "beamsearch" in args.original_model:
        # Whisper custom export with beam search contrib op
        audio = whisper.load_audio(args.audio_path)
        audio = whisper.pad_or_trim(audio)
        input_features = processor.feature_extractor([audio] * args.batch_size, return_tensors="np").input_features
        ort_inputs = {
            "input_features": input_features,
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
            "attention_mask": np.zeros((args.batch_size, args.feature_size, args.encoder_seq_len)).astype(np.int32),
        }
        exclude_list = list(ort_inputs.keys())
    elif "all" in args.original_model:
        # Whisper end-to-end ONNX model
        audio = librosa.load(args.audio_path)[0]
        audio = np.expand_dims(audio[:30 * args.sample_rate], axis=0)
        audio = audio.astype(np.uint8)
        ort_inputs = {
            "audio_stream": audio,
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
            "attention_mask": np.zeros((args.batch_size, args.feature_size, args.encoder_seq_len)).astype(np.int32),
        }
        exclude_list = list(ort_inputs.keys())
    else:
        raise Exception("Unable to auto-detect inputs for provided model")

    return set_inputs(args, ort_inputs, exclude_list)


def get_pt_inputs(args, processor, target_device):
    audio = whisper.load_audio(args.audio_path)
    audio = whisper.pad_or_trim(audio)
    
    input_features = processor.feature_extractor([audio] * args.batch_size, return_tensors="pt").input_features
    hf_inputs = {
        "inputs": input_features.to(target_device),
        "max_length": args.max_length,
        "min_length": args.min_length,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "length_penalty": args.length_penalty,
        "repetition_penalty": args.repetition_penalty,
        "attention_mask": torch.zeros((args.batch_size, args.feature_size, args.encoder_seq_len)).to(target_device, dtype=torch.int32),
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": True,
        "use_cache": True,
    }
    exclude_list = [key for key in hf_inputs if key not in {"inputs"}]
    
    return set_inputs(args, hf_inputs, exclude_list)


def set_inputs(args, input_dict, exclude_list):
    # Cast certain inputs to another dtype    
    for (k, v) in input_dict.items():
        if k in exclude_list:
            continue

        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(PRECISION[args.precision][0])
        elif isinstance(v, np.ndarray):
            input_dict[k] = v.astype(PRECISION[args.precision][1])

    return input_dict


def ort(args):
    processor = AutoProcessor.from_pretrained(args.model_name)
    sess = onnxruntime.InferenceSession(args.original_model, providers=[args.execution_provider])
    inputs = get_ort_inputs(args, processor)

    # Warm up original model
    orig_outputs = None
    for _ in range(args.warmup_runs):
        orig_outputs = sess.run(None, inputs)[0]
    
    # Benchmark original model
    start_time = time.time()
    for _ in range(args.num_runs):
        sess.run(None, inputs)
    end_time = time.time()
    latency = (end_time - start_time) / args.num_runs
    print(f"ORT original - batch size: {args.batch_size}, latency: {latency} s, throughput: {args.batch_size / latency} qps")

    sess = onnxruntime.InferenceSession(args.optimized_model, providers=[args.execution_provider])
    
    # Warm up optimized model
    opt_outputs = None
    for _ in range(args.warmup_runs):
        opt_outputs = sess.run(None, inputs)[0]
    
    # Benchmark optimized model
    start_time = time.time()
    for _ in range(args.num_runs):
        sess.run(None, inputs)
    end_time = time.time()
    latency = (end_time - start_time) / args.num_runs
    print(f"ORT optimized - batch size: {args.batch_size}, latency: {latency} s, throughput: {args.batch_size / latency} qps")


    if args.profile:
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        sess = onnxruntime.InferenceSession(args.optimized_model, sess_options, providers=[args.execution_provider])
        sess.run(None, inputs)

    # Verify outputs
    assert orig_outputs.shape == opt_outputs.shape
    parity = np.allclose(orig_outputs, opt_outputs, rtol=args.rtol, atol=args.atol)        
    print(f"Are ORT original and ORT optimized values close?", parity)
    if not parity:
        ort_diff = orig_outputs - opt_outputs
        print(f"Max difference:", np.sort(ort_diff).flatten())

    return orig_outputs, opt_outputs


def pt(args):
    target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name).to(args.device)
    inputs = get_pt_inputs(args, processor, target_device)
    
    def gen_and_dec():
        predicted_ids = model.generate(**inputs)
        if "beamsearch" in args.original_model:
            return predicted_ids.detach().cpu().numpy()
        transcription = []
        for bs in range(args.batch_size):
            for rs in range(args.num_return_sequences):
                transcription.append(
                    processor.batch_decode(predicted_ids[bs * args.num_return_sequences + rs], skip_special_tokens=True)[0]
                )
        return transcription

    # Warm up
    torch_outputs = None
    for _ in range(args.warmup_runs):
        torch_outputs = gen_and_dec()

    # Benchmark
    start_time = time.time()
    for _ in range(args.num_runs):
        gen_and_dec()
    end_time = time.time()
    latency = (end_time - start_time) / args.num_runs
    print(f"PyTorch - batch size: {args.batch_size}, latency: {latency} s, throughput: {args.batch_size / latency} qps")

    if args.profile:
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            gen_and_dec()
            print(prof.key_averages(group_by_stack_n=5).table(sort_by=args.sort_by, row_limit=1000))

    return torch_outputs


def get_args():
    parser = argparse.ArgumentParser()

    # Args for audio file
    parser.add_argument('-a', '--audio-path', type=str,
                        help="Path to audio file for E2E evaluation")
    parser.add_argument('--sample-rate', type=int, default=16000)

    # Args for beam search decoding
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--min-length', type=int, default=0)
    parser.add_argument('--max-length', type=int, default=20)
    parser.add_argument('--length-penalty', type=float, default=1.0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=3)
    parser.add_argument('--feature-size', type=int, default=80,
                        help="Known as 'feature size' or 'number of Mels'")
    parser.add_argument('--encoder-seq-len', type=int, default=3000,
                        help="Known as 'encoder sequence length' or 'number of frames'")

    # Args for ORT model
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-m', '--model-size', type=str, default="tiny")
    parser.add_argument('-p', '--precision', type=str, default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument('--original-model', type=str, default='model.onnx')
    parser.add_argument('--optimized-model', type=str, default='model_opt.onnx')

    # Args for running and evaluating
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=["cpu", "cuda"])
    parser.add_argument('-id', '--device-id', type=int, default=0,
                        help="GPU device ID when using CUDA")
    parser.add_argument('-w', '--warmup-runs', type=int, default=5)
    parser.add_argument('-r', '--num-runs', type=int, default=10)
    parser.add_argument('-s', '--seed', type=int, default=2)
    parser.add_argument('-rtol', '--rtol', type=float, default=1e-03)
    parser.add_argument('-atol', '--atol', type=float, default=1e-03)

    # Args for detailed info
    parser.add_argument('--profile', default=False, action='store_true')
    parser.add_argument('--pt-compare', default=False, action='store_true')
    parser.add_argument('--sort_by', type=str, default='self_cpu_time_total')

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")
    if args.execution_provider == "CUDAExecutionProvider":
        args.execution_provider = (args.execution_provider, {"device_id": args.device_id})

    # Set model properties
    setattr(args, 'model_name', f"openai/whisper-{args.model_size}")

    print(args.__dict__)
    return args


def main():
    args = get_args()

    ort_orig_outputs, ort_opt_outputs = ort(args)
    if args.pt_compare:
        torch_outputs = pt(args)

    # Compare ORT outputs and PyTorch outputs
    if args.pt_compare:
        if "beamsearch" in args.original_model:
            parity = np.allclose(ort_orig_outputs, torch_outputs, rtol=args.rtol, atol=args.atol)
            print(f"Are ORT original and PyTorch values close?", parity)
            if not parity:
                orig_diff = ort_orig_outputs - torch_outputs
                print(f"Max difference:", np.sort(orig_diff).flatten())

            parity = np.allclose(ort_opt_outputs, torch_outputs, rtol=args.rtol, atol=args.atol)
            print(f"Are ORT optimized and PyTorch values close?", parity)
            if not parity:
                opt_diff = ort_opt_outputs - torch_outputs
                print(f"Max difference:", np.sort(opt_diff).flatten())

        elif "all" in args.original_model:
            parity = ort_orig_outputs == torch_outputs
            print(f"Are ORT original and PyTorch values close?", parity)
            if not parity:
                print("ORT original outputs:", ort_orig_outputs)
                print("PyTorch outputs:", torch_outputs)

            parity = ort_opt_outputs == torch_outputs
            print(f"Are ORT optimized and PyTorch values close?", parity)
            if not parity:
                print("ORT optimized outputs:", ort_opt_outputs)
                print("PyTorch outputs:", torch_outputs)


if __name__ == '__main__':
    main()
