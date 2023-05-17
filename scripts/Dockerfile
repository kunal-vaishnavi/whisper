FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Define environment info
ARG WHISPER_ORT_REPO=https://github.com/kunal-vaishnavi/whisper
ARG WHISPER_ORT_BRANCH=ort
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
RUN pip install torch torchaudio torchvision --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
RUN pip install optimum onnx coloredlogs psutil py3nvml onnxconverter_common ffmpeg-python librosa
RUN pip install --upgrade optimum
# Install specific NumPy version to satisfy numba and ORT
RUN pip install numpy==1.23.5

# Install necessary software through pip + git
RUN pip install git+https://github.com/microsoft/onnxruntime-extensions.git
RUN pip install git+https://github.com/openai/whisper.git
# Reinstall transformers from source
RUN pip uninstall -y transformers
RUN pip install git+https://github.com/kunal-vaishnavi/transformers.git@ort

# Install ORT
ARG ORT_WHL=ort_nightly_gpu-1.16.0.dev20230516003-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
ADD ${ORT_WHL} .
RUN pip install ${ORT_WHL}

# RUN pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly
# RUN pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-extensions

# Add files to home directory
WORKDIR home/
RUN wget https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/tutorials/whisper_e2e.py
RUN wget https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/1272-141231-0002.mp3
RUN wget https://raw.githubusercontent.com/kunal-vaishnavi/whisper/ort/scripts/benchmark.py
RUN wget https://raw.githubusercontent.com/kunal-vaishnavi/whisper/ort/scripts/parity.py

RUN git clone ${WHISPER_ORT_REPO} --branch ${WHISPER_ORT_BRANCH} --single-branch
RUN git clone ${WHISPER_CPP_REPO} --branch ${WHISPER_CPP_BRANCH} --single-branch
RUN cd whisper.cpp && make
