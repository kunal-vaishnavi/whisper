#######################################################
# Save Whisper model in ONNX format
#
# Run script as python3 scripts/whisper_hf_to_onnx.py
#######################################################

import argparse
import os
import shutil
import subprocess

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

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
    model_id = f"openai/whisper-{args.size}.en"
    whisper_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        from_transformers=True, 
        use_io_binding=True,
    )
    cached_path = whisper_model.model_save_dir

    # Create cache folder to save ONNX models
    output_path = os.path.join(args.path, args.size)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Save separate encoder/decoder ONNX models
    for fle in os.listdir(cached_path):
        shutil.copy(os.path.join(cached_path, fle), output_path)

    # Save combined encoder/decoder ONNX models
    subprocess.run(["python3", "-m", "optimum.exporters.onnx", "--model", model_id, output_path])


if __name__ == '__main__':
    main()
