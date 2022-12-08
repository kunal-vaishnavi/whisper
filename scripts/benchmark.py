############################################################################################
# Benchmark Whisper model
#
# For up-to-date benchmarking, run this script as follows:
# python3 whisper_with_ort.py --batch_size <batch-size> --device <device> --engine <engine>
# Append --path <path> for ORT engine, --precision <precision> for PyTorch engine
#
# To run PyTorch FP16 and/or PyTorch 2.0:
# 1) git clone https://github.com/huggingface/transformers && cd transformers
# 2) Go to src/transformers/pipelines/automatic_speech_recognition.py
# 3) Mmake the following changes:
#
# 3a) To run PyTorch FP16: 
# Before:
# processed = self.feature_extractor(
#    inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
# )
#
# After:
# processed = self.feature_extractor(
#    inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
# )
# import torch
# processed['input_features'] = processed['input_features'].to(torch.float16)
#
# 3b) To run PyTorch 2.0:
# Before:
# if self.model.__class__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values():
#     self.type = "seq2seq"
#
# After:
# import torch
# if self.model.__class__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values() or isinstance(self.model, torch._dynamo.eval_frame.OptimizedModule):
#     self.type = "seq2seq"
#
# 4) Go back to parent folder where setup.py is
# 5) pip install -e .
#
# Note: Comment out the changes when not running PyTorch FP16 and/or PyTorch 2.0.
############################################################################################

import argparse
import gc
import numpy as np
import os
import onnxruntime
import time
import torch
import whisper

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import pipeline as pt_pipeline
from transformers.onnx.utils import get_preprocessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.pipelines import pipeline as ort_pipeline
from onnxruntime.transformers.benchmark_helper import measure_memory

MODEL_NAME = "openai/whisper-tiny.en"

def get_ort(device, directory):
    processor = get_preprocessor(MODEL_NAME)
    pipeline = lambda *args, **kwargs: ort_pipeline(*args, **kwargs)
    
    if directory is None:
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME, 
            from_transformers=True,
            use_io_binding=True,
        ).to(device)
    else:
        assert os.path.exists(directory)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            directory,
            use_io_binding=True,
        ).to(device)
    
    return (processor, model, pipeline)

def get_torch(device, precision):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=(torch.float16 if precision == "fp16" else torch.float32)
    ).to(device)
    pipeline = lambda *args, **kwargs: pt_pipeline(*args, **kwargs)
    return (processor, model, pipeline)    

def get_vars(args):
    if args.engine == 'ort':
        return get_ort(args.device, args.path)
    if args.engine == 'pt':
        return get_torch(args.device, args.precision)
    raise NotImplementedError('Invalid engine specified')

# Data generator from optimum test cases
def generate_data():
    np.random.seed(10)
    t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
    return audio_data.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '-f',
        '--path',
        default=None,
        help='Path to folder of Whisper ONNX models',
    )
    group.add_argument(
        '-p',
        '--precision',
        choices=['fp16', 'fp32'],
        help='Precision of Whisper PyTorch models',
    )

    parser.add_argument(
        '-d',
        '--device',
        required=False,
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        required=False,
        type=int,
        default=1,
    )

    parser.add_argument(
        '-e',
        '--engine',
        required=False,
        type=str,
        default='ort',
        choices=['ort', 'pt'],
    )

    parser.add_argument(
        '-pt2',
        '--pytorch2',
        required=False,
        action='store_true',
        help='Whether to use PyTorch 2.0 (i.e. whether to use torch.compile(model) or not)',
    )
    parser.set_defaults(pytorch2=False)

    parser.add_argument(
        '-v',
        '--verbose',
        required=False,
        action='store_true',
        help="Whether to print information (e.g. outputs, verifications)",
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.__dict__)
    torch.backends.cudnn.benchmark = True
    processor, model, pipeline = get_vars(args)
    if args.pytorch2:
        model = torch.compile(model)

    # Load audio file
    audio = whisper.load_audio('tests/jfk.flac')
    audio = whisper.pad_or_trim(audio)

    # Calculate log-Mel spectrogram through Whisper repo or through HuggingFace
    # 1) Repo
    mel_spectrogram = whisper.log_mel_spectrogram(audio)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    mel_spectrogram = np.tile(mel_spectrogram, (args.batch_size, 1, 1))
    # 2) HuggingFace
    features = processor.feature_extractor([audio] * args.batch_size, return_tensors="pt")
    
    # Move audio info to torch and device, then assert shapes and values
    mel_spectrogram = torch.from_numpy(mel_spectrogram).to(args.device)
    features = features.to(args.device)
    if args.verbose:
        print("log-Mel spectrogram:", mel_spectrogram.shape)
        assert features['input_features'].shape == mel_spectrogram.shape
        assert torch.allclose(mel_spectrogram, features['input_features'], rtol=1e-3, atol=1e-3)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=(-1 if args.device == 'cpu' else 0),
    )
    torch.cuda.synchronize()
    start_time = time.time()
    outputs = pipe([audio] * args.batch_size)
    torch.cuda.synchronize()
    end_time = time.time()
    latency = end_time - start_time
    print(f"Batch size = {args.batch_size}, latency = {latency} s, throughput = {args.batch_size / latency} queries/s")

    gc.collect()
    torch.cuda.empty_cache()
    measure_memory(is_gpu=(args.device == 'cuda'), func=lambda: pipe([audio] * args.batch_size))

    if args.verbose:
        print(outputs)

if __name__ == '__main__':
    main()
