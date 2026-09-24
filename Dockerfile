FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgdal-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/app/requirements.txt /app/requirements.txt

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install -r /app/requirements.txt

COPY backend /app/backend
COPY models /app/models

RUN mkdir -p /app/backend/outputs

EXPOSE 7860

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "7860"]
