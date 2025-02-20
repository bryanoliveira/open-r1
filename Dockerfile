# Build
# docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg WANDB_KEY=<wandb> --build-arg HF_TOKEN=<hf> -f Dockerfile -t aluno_bryan-openr1:latest .
# Run (local)
# sudo docker run --gpus '"device=0"' -v ~/.cache/huggingface:/root/.cache/huggingface -e MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B aluno_bryan-openr1:latest
# Run (dgx)
# docker run --gpus '"device=5"' -v /raid/bryan/huggingface:/root/.cache/huggingface -e MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -e MODEL_ARGS='dtype=float16,max_model_length=32768,gpu_memory_utilisation=0.9' aluno_bryan-openr1:latest

# Use CUDA 12.4 base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Create and activate virtual environment
RUN uv venv /opt/venv --python python3.11
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN uv pip install --upgrade pip --link-mode=copy && \
    uv pip install vllm==0.7.2 setuptools wheel --link-mode=copy

# Copy project files
WORKDIR /app
COPY . .

# Install project dependencies
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy --no-build-isolation

# Default environment variables for evaluation
ENV MODEL_ARGS="dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8"
ENV TASK="aime24"
ENV OUTPUT_DIR="/app/data/evals"

# Entrypoint that runs lighteval with the provided model
ENTRYPOINT ["sh", "-c", "lighteval vllm pretrained=${MODEL},${MODEL_ARGS} \"custom|${TASK}|0|0\" --custom-tasks src/open_r1/evaluate.py --use-chat-template --output-dir ${OUTPUT_DIR}/${MODEL}"]