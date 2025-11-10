FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# System deps for OpenCV, pycolmap, etc.
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils \
        build-essential git wget ffmpeg libgl1 libglib2.0-0 libboost-all-dev \
        && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . .

# Torch first so CUDA wheels come from PyTorch index
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

ENV SAM2_BUILD_CUDA=0
RUN pip install 'git+https://github.com/facebookresearch/sam2.git'

# Build Sim3 CUDA extension once
RUN python3.10 setup.py install

# Optional: bake commonly used checkpoints into the image
RUN bash ./scripts/download_weights.sh
RUN mkdir -p checkpoints && \
    wget -nc -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

CMD ["python3.10", "pi_long.py", "--help"]
