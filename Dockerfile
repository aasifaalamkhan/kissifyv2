# Use the official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for non-buffered Python output and CUDA home
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies:
# - git for cloning repositories (like IP-Adapter)
# - ffmpeg for video processing (used by diffusers export_to_video)
# - build-essential for compiling some Python packages (e.g., controlnet-aux, or direct Git installs)
# - python3-dev for Python header files, needed for compiling Python extensions
# - curl for HEALTHCHECK (if used)
# We use --no-install-recommends to keep the image size smaller and clean up apt cache.
RUN echo "--- Updating apt packages and installing system dependencies ---" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        python3-dev \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "--- System dependencies installed ---"

# --- Optional: Install Git LFS if your models might use it ---
# Uncomment if needed for specific Hugging Face models
# RUN echo "--- Installing Git LFS ---" && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends git-lfs && \
#     git lfs install && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     echo "--- Git LFS installed ---"

# Copy the requirements file and install Python packages.
# --no-cache-dir prevents pip from storing downloaded packages, reducing image size.
# --upgrade pip ensures pip is up-to-date, setuptools and wheel are also good to update.
COPY requirements.txt ./requirements.txt
RUN echo "--- Installing Python packages from requirements.txt ---" && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    echo "--- Python packages installed ---"

# --- OPTIONAL: Pre-fetch Hugging Face models during build ---
# Uncomment this section for production builds to reduce cold start times.
# Ensure you pass your Hugging Face token as a build secret or environment variable during the build process.
# Example for RunPod `runpod.yaml` (simplified):
# build_args:
#   - name: HUGGING_FACE_HUB_TOKEN
#     value: runpod_secret:YOUR_HF_TOKEN_SECRET_NAME
# RUN --mount=type=secret,id=huggingface_token,target=/run/secrets/huggingface_token \
#     export HF_HOME="/app/hf_cache" && \
#     mkdir -p "${HF_HOME}" && \
#     echo "--- Pre-fetching Hugging Face models ---" && \
#     python -c " \
# import os; \
# from huggingface_hub import snapshot_download; \
# token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or open('/run/secrets/huggingface_token').read().strip(); \
# hf_cache_dir = os.environ.get('HF_HOME'); \
# print('Pre-fetching SG161222/Realistic_Vision_V5.1_noVAE...'); \
# snapshot_download(repo_id='SG161222/Realistic_Vision_V5.1_noVAE', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'SG161222/Realistic_Vision_V5.1_noVAE'), resume_download=True, token=token); \
# print('Pre-fetching guoyww/animatediff-motion-module-v3...'); \
# snapshot_download(repo_id='guoyww/animatediff-motion-module-v3', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'guoyww/animatediff-motion-module-v3'), resume_download=True, token=token); \
# print('Pre-fetching lllyasviel/control_v11p_sd15_openpose...'); \
# snapshot_download(repo_id='lllyasviel/control_v11p_sd15_openpose', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'lllyasviel/control_v11p_sd15_openpose'), resume_download=True, token=token); \
# print('Pre-fetching lllyasviel/control_v11f1p_sd15_depth...'); \
# snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'lllyasviel/control_v11f1p_sd15_depth'), resume_download=True, token=token); \
# print('Pre-fetching h94/IP-Adapter...'); \
# snapshot_download(repo_id='h94/IP-Adapter', allow_patterns=['models/image_encoder/*', 'ip-adapter_sd15.bin'], local_dir=os.path.join(hf_cache_dir, 'h94/IP-Adapter'), resume_download=True, token=token); \
# print('Model pre-fetching complete.'); \
# " && \
#     echo "--- Pre-fetched models successfully ---"

# Copy your main application script into the container.
COPY main.py ./main.py

# Expose port 3000 for the health check server.
EXPOSE 3000

# Add a basic health check to inform Docker/RunPod about container readiness
# Requires 'curl' to be installed in system dependencies.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/healthz || exit 1

# This is the command that RunPod will execute when the container starts.
CMD ["python", "main.py"]
