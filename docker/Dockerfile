FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Add uv to the PATH
ENV PATH="/root/.local/bin:${PATH}"

# CHANGED: Match the /app directory used in compose and devcontainer
WORKDIR /app

# Install prerequisites, Python 3.11, and the venv module
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    libegl1 \
    libosmesa6-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python command
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Download and install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

CMD ["sleep", "infinity"]