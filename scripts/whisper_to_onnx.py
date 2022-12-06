import argparse
import onnx
import os
import shutil
import torch
import whisper

def export_model(model, model_size, model_args, model_path, input_names, output_names, dynamic_axes, weight_fname=None, opset=15):
    too_large = model_size in {'small', 'medium', 'large'} # small: ~2GB, medium: ~5GB, large: ~10GB

    torch.onnx.export(
        model,
        model_args,
        f=model_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset
    )

    if too_large:
        onnx_model = onnx.load(model_path)
        onnx.save_model(
            onnx_model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=weight_fname,
            convert_attribute=False
        )

def to_onnx(model, model_size, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_path = os.path.join(output_path, "encoder.onnx")
    decoder_path = os.path.join(output_path, "decoder.onnx")

    # Export Whisper encoder model to ONNX
    export_model(
        model.encoder,
        model_size=model_size,
        model_args=(torch.randn(1, 80, 3000).to(device=device)),
        model_path=encoder_path,
        input_names=["mel_spectrogram"],
        output_names=["audio_features"],
        dynamic_axes={ "mel_spectrogram": {0: "batch_size", 1: "n_mels", 2: "n_ctx"} },
        weight_fname="encoder_weights.pb"
    )

    # Export Whisper decoder model to ONNX
    export_model(
        model.decoder,
        model_size=model_size,
        model_args=(
            torch.randn(1, 1).to(device=device).long(),
            torch.randn(1, 384, 384).to(device=device)
        ),
        model_path=decoder_path,
        input_names=["text_tokens", "audio_features"],
        output_names=["logits"],
        dynamic_axes={ "text_tokens": {0: "batch_size", 1: "n_ctx"}, "audio_features": {0: "batch_size", 1: "n_mels", 2: "n_audio_ctx"} },
        weight_fname="decoder_weights.pb"
    )

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-s',
        '--size',
        required=False,
        default='tiny',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Size of Whisper model to load'
    )

    parser.add_argument(
        '-p',
        '--path',
        required=False,
        default='./onnx',
        help='Destination folder to save ONNX models',
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    whisper_model = whisper.load_model(args.size)

    output_path = os.path.join(args.path, args.size)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    to_onnx(whisper_model, args.size, output_path)

if __name__ == '__main__':
    main()