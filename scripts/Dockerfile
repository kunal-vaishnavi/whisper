FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Define environment info
ARG WHISPER_REPO=https://github.com/kunal-vaishnavi/whisper
ARG WHISPER_BRANCH=ort
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary software through apt-get
RUN apt-get update --fix-missing
RUN apt-get install -y sudo vim git bash python3 python3-dev python3-pip python3-wheel ffmpeg
RUN cd /usr/local/bin && ln -s /usr/bin/python3 python && ln -s /usr/bin/pip3 pip;
RUN apt-get update

# Install necessary software through pip
RUN pip install torch torchaudio torchvision --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
RUN pip install numpy==1.23.3 transformers optimum onnx coloredlogs psutil py3nvml onnxconverter_common onnxruntime-gpu whisper.ai ffmpeg-python
RUN pip install --upgrade transformers optimum

# Get Whisper repo to have access to whisper_to_onnx.py and whisper_with_ort.py
WORKDIR home/
RUN git clone ${WHISPER_REPO}
RUN cd whisper && git checkout ${WHISPER_BRANCH} && pip install -e .

# Install hf_onnx/ and onnx/tiny/ folders
RUN cd whisper && python3 -m optimum.exporters.onnx --model openai/whisper-tiny.en hf_onnx/
RUN cd whisper && python3 whisper_to_onnx.py





#######################################################
# Integrating Whisper into comprehensive Docker image:
#######################################################
# FROM kunalva/sd_bench:latest

# ARG WHISPER_REPO=https://github.com/kunal-vaishnavi/whisper
# ARG WHISPER_BRANCH=ort

# # Install necessary software
# RUN apt-get install ffmpeg
# RUN pip install --upgrade transformers optimum whisper.ai ffmpeg-python

# # Get Whisper repo to have access to whisper_to_onnx.py and whisper_with_ort.py
# RUN git clone ${WHISPER_REPO}
# RUN cd whisper && git checkout ${WHISPER_BRANCH} && pip install -e .

# # Install hf_onnx/ and onnx/tiny/ folders
# RUN cd whisper && python3 -m optimum.exporters.onnx --model openai/whisper-tiny.en hf_onnx/
# RUN cd whisper && python3 whisper_to_onnx.py
