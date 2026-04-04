FROM python:3.11-slim

# System dependencies:
#   ffmpeg        — required by faster-whisper for audio decoding
#   libasound2    — required by Azure Speech SDK
#   libssl-dev    — required by Azure Speech SDK
#   gcc / g++     — required to compile chromadb native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libasound2 \
    libssl-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and static assets
COPY app/ ./app/
COPY static/ ./static/
COPY knowledge_base/ ./knowledge_base/
COPY data/ ./data/

# HuggingFace / Torch model cache — mount a volume here to persist downloaded models
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
