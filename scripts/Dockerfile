FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Define environment info
ARG WHISPER_CPP_REPO=https://github.com/ggerganov/whisper.cpp
ARG WHISPER_CPP_BRANCH=master
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary software through apt-get
RUN apt-get update --fix-missing
RUN apt-get install -y sudo vim git bash wget python3 python3-dev python3-pip python3-wheel ffmpeg
RUN cd /usr/local/bin && ln -s /usr/bin/python3 python && ln -s /usr/bin/pip3 pip
RUN apt-get update

# Install necessary software through pip
RUN pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip install optimum onnx coloredlogs psutil py3nvml onnxconverter_common ffmpeg-python librosa

# Install nightly ORT
ARG ORT_VERSION=1.16.0.dev20230901001
RUN pip install ort-nightly-gpu=={ORT_VERSION} --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/

# Install necessary software through pip + git
RUN pip install git+https://github.com/openai/whisper.git
RUN pip install git+https://github.com/microsoft/onnxruntime-extensions.git
RUN pip uninstall -y transformers && pip install git+https://github.com/kunal-vaishnavi/transformers.git@ort

# Install specific NumPy version to satisfy numba and ORT
RUN pip uninstall -y numpy && pip install numpy==1.23.5

# Add files to home directory
WORKDIR home/
RUN wget https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/tutorials/whisper_e2e.py
RUN wget https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/1272-141231-0002.mp3

RUN git clone ${WHISPER_CPP_REPO} --branch ${WHISPER_CPP_BRANCH} --single-branch
RUN cd whisper.cpp && make
