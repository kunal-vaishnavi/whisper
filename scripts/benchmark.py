##########################################################################################
# Benchmark Whisper model in ONNX Runtime
#
# For up-to-date benchmarking, run this script as follows:
# python3 whisper_with_ort.py --modes hf-pipe --batch_size <batch-size> --device <device>
##########################################################################################

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

# Data generator from optimum test cases
def generate_data():
    np.random.seed(10)
    t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
    return audio_data.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-p',
        '--path',
        required=False,
        type=str,
        default='onnx/tiny/',
        help='Path to folder of Whisper ONNX models'
    )

    parser.add_argument(
        '-m',
        '--modes',
        required=False,
        nargs='+',
        default=['hf-pipe'],
        help="Options are: ['hf-pipe', 'hf-token', 'hf-logits', 'hf-ort-logits', 'ort']",
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
        '-v',
        '--verbose',
        required=False,
        action='store_true',
        help="Whether to print information (e.g. outputs, verifications)",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        '--decoder_sequence_length',
        required=False,
        type=int,
        default=20,
    )

    args = parser.parse_args()
    
    if args.engine == 'ort':
        device_to_ep = {'cuda': 'CUDAExecutionProvider', 'cpu': "CPUExecutionProvider"}
        setattr(args, 'providers', [device_to_ep[args.device]])
    return args

def main():
    args = parse_args()
    print(args.__dict__)

    # Variables used in the different run options
    model_id = "openai/whisper-tiny.en"
    if args.engine == 'ort':
        processor = get_preprocessor(model_id)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, from_transformers=True, use_io_binding=True).to(args.device)
        pipeline = lambda *args, **kwargs: ort_pipeline(*args, **kwargs)
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(args.device)
        pipeline = lambda *args, **kwargs: pt_pipeline(*args, **kwargs)
    decoder_start_token_id = model._get_decoder_start_token_id()
    decoder_inputs = {"decoder_input_ids": torch.ones((args.batch_size, args.decoder_sequence_length), dtype=torch.long) * decoder_start_token_id}

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


    if 'hf-pipe' in args.modes:
        # Option 1: Pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=(-1 if args.device == 'cpu' else 0),
        )
        start_time = time.time()
        outputs = pipe([audio] * args.batch_size)
        end_time = time.time()
        latency = end_time - start_time
        print(f"Batch size = {args.batch_size}, latency = {latency} s, throughput = {args.batch_size / latency} queries/s")

        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == 'cuda'), func=lambda: pipe([audio] * args.batch_size))

        if args.verbose:
            print(outputs) # outputs is of the form [{'text': '<translation>'}, ...]


    if 'hf-token' in args.modes:
        # Option 2: No pipeline
        start_time = time.time()
        ids = model.generate(**features, num_beams=5)
        outputs = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        end_time = time.time()
        latency = end_time - start_time
        print(f"Batch size = {args.batch_size}, latency = {latency} s, throughput = {args.batch_size / latency} queries/s")

        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == 'cuda'), func=lambda: pipe([audio] * args.batch_size))

        if args.verbose:
            print(ids)
            print(outputs) # outputs is of the form ['<translation>', ...]


    if 'hf-logits' in args.modes:
        # Option 3: Manual calculation using logits
        outputs = model(**features, **decoder_inputs)
        if args.verbose:
            print("Logits through combined encoder/decoder:", outputs.logits.shape)
            # ids = outputs.logits.argmax(dim=-1)
            # probs = outputs.logits.softmax(dim=-1)
            # outputs = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            # print(outputs)
            
    
    if 'hf-ort-logits' in args.modes:
        # Option 4: Using native ORT but with HuggingFace's ONNX model
        logits_shape, matmul_711_shape = [args.batch_size, args.decoder_sequence_length, 51864], [args.batch_size, 1500, 384]
        input_features_ort = onnxruntime.OrtValue.ortvalue_from_numpy(features['input_features'].detach().cpu().numpy(), args.device, 0)
        decoder_input_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(decoder_inputs['decoder_input_ids'].detach().cpu().numpy(), args.device, 0)
        logits_ort = onnxruntime.OrtValue.ortvalue_from_shape_and_type(logits_shape, input_features_ort.numpy().dtype, args.device, 0)
        matmul_711_ort = onnxruntime.OrtValue.ortvalue_from_shape_and_type(matmul_711_shape, input_features_ort.numpy().dtype, args.device, 0)

        session = onnxruntime.InferenceSession(os.path.join("hf_onnx", "model.onnx"), providers=args.providers)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input('input_features', input_features_ort)
        io_binding.bind_ortvalue_input('decoder_input_ids', decoder_input_ids_ort)
        io_binding.bind_ortvalue_output('logits', logits_ort)
        io_binding.bind_ortvalue_output('onnx::MatMul_711', matmul_711_ort)
        session.run_with_iobinding(io_binding)
        print("ORT inference session is complete")


    if args.verbose and 'hf-logits' in args.modes and 'hf-onnx-logits' in args.modes:
        print("Logits through combined encoder/decoder:", outputs.logits.shape)
        assert outputs.logits.shape == logits_ort.numpy().shape
        assert np.allclose(outputs.logits.detach().cpu().numpy(), logits_ort.numpy(), rtol=1e-3, atol=1e-3)
        print("Logits calculated through HuggingFace's ONNX model and through HuggingFace's abstracted implementation are equal")


    if 'ort' in args.modes:
        # Option 5: Use converter from whisper_to_onnx.py and native ORT
        audio_features_shape = [args.batch_size, 1500, 384]

        # Run encoder through ORT
        mel_spectrogram_ort = onnxruntime.OrtValue.ortvalue_from_numpy(mel_spectrogram.detach().cpu().numpy(), args.device, 0)
        audio_features_ort = onnxruntime.OrtValue.ortvalue_from_shape_and_type(audio_features_shape, mel_spectrogram.detach().cpu().numpy().dtype, args.device, 0)

        session = onnxruntime.InferenceSession(os.path.join(args.path, "encoder.onnx"), providers=args.providers)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input('mel_spectrogram', mel_spectrogram_ort)
        io_binding.bind_ortvalue_output('audio_features', audio_features_ort)

        start_time = time.time()
        session.run_with_iobinding(io_binding)
        print(f"Encoder took {time.time() - start_time} seconds")

        # Run decoder through ORT
        text_tokens = np.ones((args.batch_size, args.decoder_sequence_length), dtype=np.int64) * decoder_start_token_id
        logits_shape = [args.batch_size, args.decoder_sequence_length, 51865]

        text_tokens_ort = onnxruntime.OrtValue.ortvalue_from_numpy(text_tokens, args.device, 0)
        logits_ort = onnxruntime.OrtValue.ortvalue_from_shape_and_type(logits_shape, audio_features_ort.numpy().dtype, args.device, 0)

        session = onnxruntime.InferenceSession(os.path.join(args.path, "decoder.onnx"), providers=args.providers)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input('text_tokens', text_tokens_ort)
        io_binding.bind_ortvalue_input('audio_features', audio_features_ort)
        io_binding.bind_ortvalue_output('logits', logits_ort)

        start_time = time.time()
        session.run_with_iobinding(io_binding)
        print(f"Decoder took {time.time() - start_time} seconds")

        if args.verbose:
            print("Logits through separate encoder/decoder:", logits_ort.numpy().shape)
            # ids = logits_ort.numpy().argmax(axis=-1)
            # preds = logits_torch.softmax(dim=-1)
            # outputs = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            # print(outputs)


if __name__ == '__main__':
    main()
