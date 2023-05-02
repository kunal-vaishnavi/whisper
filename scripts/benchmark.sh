#!/bin/bash

#########################
# Starting Docker image:
# docker run --gpus all -it kunalva/whisper_bench:latest
#########################

################
# E2E examples:
################

# 1) "HF + PT", FP32, pipeline
# python3 benchmark.py \
#     --benchmark-type "HF + PT" \
#     --hf-api pipeline \
#     --audio-path 1272-141231-0002.mp3 \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 2) "HF + PT", FP32, gen-and-dec
# python3 benchmark.py \
#     --benchmark-type "HF + PT" \
#     --hf-api gen-and-dec \
#     --audio-path 1272-141231-0002.mp3 \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 3) "HF + PT2", FP32, pipeline
# python3 benchmark.py \
#     --benchmark-type "HF + PT2" \
#     --hf-api pipeline \
#     --audio-path whisper/tests/jfk.flac \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 4) "HF + PT2", FP32, gen-and-dec
# python3 benchmark.py \
#     --benchmark-type "HF + PT2" \
#     --hf-api gen-and-dec \
#     --audio-path whisper/tests/jfk.flac \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 5) "HF + ORT", FP32, pipeline
# python3 benchmark.py \
#     --benchmark-type "HF + ORT" \
#     --hf-api pipeline \
#     --audio-path whisper/tests/jfk.flac \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 6) "HF + ORT", FP32, gen-and-dec
# python3 benchmark.py \
#     --benchmark-type "HF + ORT" \
#     --hf-api gen-and-dec \
#     --ort-model-path wtiny_fp32/openai/whisper-tiny_beamsearch.onnx
#     --audio-path whisper/tests/jfk.flac \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 7) "ORT only", FP32, E2E
# python3 benchmark.py \
#     --benchmark-type ORT \
#     --ort-model-path wtiny_fp32/openai/whisper-tiny_all.onnx \
#     --audio-path 1272-141231-0002.mp3 \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

#################################
# Individual component examples:
#################################

# 1) "Whisper encoder" from optimum export, FP32
# python3 benchmark.py \
#     --benchmark-type "ORT" \
#     --ort-model-path hf-ort-optimum-whisper-tiny/fp32/encoder_model.onnx \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 2) "Whisper decoder" from optimum export, FP32
# python3 benchmark.py \
#     --benchmark-type "ORT" \
#     --ort-model-path hf-ort-optimum-whisper-tiny/fp32/decoder_model.onnx \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 3) "Whisper decoder-with-past" from optimum export, FP32
# python3 benchmark.py \
#     --benchmark-type "ORT" \
#     --ort-model-path hf-ort-optimum-whisper-tiny/fp32/decoder_with_past_model.onnx \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 4) "Whisper encoder-decoder-init" from custom export, FP32
# python3 benchmark.py \
#     --benchmark-type "ORT" \
#     --ort-model-path wtiny_fp32/openai/whisper-tiny_encoder_decoder_init_fp32.onnx \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 5) "Whisper decoder-with-past" from custom export, FP32
# python3 benchmark.py \
#     --benchmark-type "ORT" \
#     --ort-model-path wtiny_fp32/openai/whisper-tiny_decoder_fp32.onnx \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

# 6) "Whisper with beam search op" from custom export, FP32
# python3 benchmark.py \
#     --benchmark-type "ORT" \
#     --ort-model-path wtiny_fp32/openai/whisper-tiny_beamsearch.onnx \
#     --precision fp32 \
#     --model-size tiny \
#     --batch-size 2 \
#     --device cpu \

###################
# Parity examples:
###################

# 1) "Whisper with beam search op" from custom export, FP32
# python3 parity.py \
#     --audio-path 1272-141231-0002.mp3 \
#     --original-model wtiny_fp32/openai/whisper-tiny_beamsearch.onnx \
#     --optimized-model wtiny_opt_fp32/openai/whisper-tiny_beamsearch.onnx \
#     --device cpu

# 2) "Whisper with beam search op" from custom export, FP32, PyTorch compare
# python3 parity.py \
#     --audio-path 1272-141231-0002.mp3 \
#     --original-model wtiny_fp32/openai/whisper-tiny_beamsearch.onnx \
#     --optimized-model wtiny_opt_fp32/openai/whisper-tiny_beamsearch.onnx \
#     --device cpu \
#     --pt-compare
