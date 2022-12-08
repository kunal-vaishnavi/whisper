#####################################################################
# Convert Whisper model in ONNX from FP32 to FP16
#
# Run script after python3 scripts/whisper_hf_to_onnx.py
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

def setup_dir(folder):
    fp16_folder_name, fp32_folder_name = "fp16", "fp32"
    os.rename(folder, fp32_folder_name)
    os.makedirs(folder, exist_ok=True)
    shutil.copytree(fp32_folder_name, fp16_folder_name)
    shutil.move(fp32_folder_name, folder)
    output_dir = shutil.move(fp16_folder_name, folder)
    return output_dir

def main():
    args = parse_args()
    output_dir = setup_dir(args.folder)

    for fle in os.listdir(output_dir):
        if ".onnx" in fle:
            onnx_model_path = os.path.join(output_dir, fle)
            m = optimize_model(
                onnx_model_path,
                model_type="bert",
                num_heads=0,
                hidden_size=0,
                opt_level=0,
                optimization_options=None,
                use_gpu=False,
            )

            m.convert_float_to_float16()
            onnx.save_model(m.model, onnx_model_path)

if __name__ == '__main__':
    main()
