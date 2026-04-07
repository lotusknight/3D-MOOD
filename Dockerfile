ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    TORCH_CUDA_ARCH_LIST=8.6 \
    CUDA_HOME=/usr/local/cuda \
    MOOD_HOST=0.0.0.0 \
    MOOD_PORT=8000

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    build-essential \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel

# Pin a tested CUDA/PyTorch stack, then install Vis4D + CUDA ops.
RUN python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install vis4d==1.0.0

RUN curl -L --retry 5 --retry-delay 5 \
    -o /tmp/vis4d_cuda_ops.tar.gz \
    https://codeload.github.com/SysCV/vis4d_cuda_ops/tar.gz/refs/heads/main \
    && mkdir -p /tmp/vis4d_cuda_ops \
    && tar -xzf /tmp/vis4d_cuda_ops.tar.gz -C /tmp/vis4d_cuda_ops --strip-components=1 \
    && sed -i 's/if torch.cuda.is_available() and CUDA_HOME is not None:/if CUDA_HOME is not None:/' /tmp/vis4d_cuda_ops/setup.py \
    && FORCE_CUDA=1 python -m pip install /tmp/vis4d_cuda_ops \
    --no-build-isolation \
    --no-cache-dir \
    && rm -rf /tmp/vis4d_cuda_ops /tmp/vis4d_cuda_ops.tar.gz

RUN python -m pip install -e .

EXPOSE 8000

CMD ["python", "scripts/serve_fastapi.py", "--host", "0.0.0.0", "--port", "8000"]
