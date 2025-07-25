FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Create app directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install --upgrade huggingface_hub \
 && rm -rf /root/.cache/pip  # Clean up pip cache

# Preload models (optional but helpful to reduce cold start time)
RUN python3 download_models.py  # Fail the build if model download fails

# RunPod serverless expects this exact entrypoint
CMD ["python3", "handler.py"]
