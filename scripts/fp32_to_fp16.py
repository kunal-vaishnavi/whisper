#####################################################################
# Convert Whisper model in ONNX from FP32 to FP16
#
# Run script after python3 scripts/hf_to_onnx.py
# Run script as python3 scripts/fp32_to_fp16.py -f <path-to-folder>
#####################################################################

import argparse
import os
import shutil
import onnx
from onnxruntime.transformers.optimizer import optimize_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--folder',
        required=False,
        type=str,
        default='./onnx/tiny',
        help="Root directory of the Whisper ONNX files",
    )
    
    args = parser.parse_args()
    return args

def setup_dir(parent_folder):
    fp16_folder_name, fp32_folder_name = "fp16", "fp32"
    os.rename(parent_folder, fp32_folder_name)
    os.makedirs(parent_folder, exist_ok=True)
    shutil.copytree(fp32_folder_name, fp16_folder_name)
    fp32_path = shutil.move(fp32_folder_name, parent_folder)
    fp16_path = shutil.move(fp16_folder_name, parent_folder)
    return fp16_path, fp32_path

def main():
    args = parse_args()
    fp16_path, fp32_path = setup_dir(args.folder)

    for fle in os.listdir(fp32_path):
        if ".onnx" in fle:
            fp16_model_path = os.path.join(fp16_path, fle)
            fp32_model_path = os.path.join(fp32_path, fle)
            m = optimize_model(
                fp32_model_path,
                model_type="bert",
                num_heads=0,
                hidden_size=0,
                opt_level=0,
                optimization_options=None,
                use_gpu=False,
            )
            onnx.save_model(m.model, fp32_model_path)
            m.convert_float_to_float16()
            onnx.save_model(m.model, fp16_model_path)

if __name__ == '__main__':
    main()
