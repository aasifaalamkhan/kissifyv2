FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

RUN N_RETRIES=5; \
    for i in $(seq 1 $N_RETRIES); do \
        apt-get update && break || { \
            if [ $i -lt $N_RETRIES ]; then \
                echo "apt-get update failed, retrying in 5 seconds..."; \
                sleep 5; \
            else \
                echo "apt-get update failed after $N_RETRIES attempts. Exiting."; \
                exit 1; \
            fi; \
        }; \
    done && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        python3-dev \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Uncomment the following section to pre-fetch Hugging Face models during build
# RUN --mount=type=secret,id=huggingface_token,target=/run/secrets/huggingface_token \
#     export HF_HOME="/app/hf_cache" && \
#     mkdir -p "${HF_HOME}" && \
#     python -c " \
# import os; \
# from huggingface_hub import snapshot_download; \
# token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or open('/run/secrets/huggingface_token').read().strip(); \
# hf_cache_dir = os.environ.get('HF_HOME'); \
# snapshot_download(repo_id='SG161222/Realistic_Vision_V5.1_noVAE', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'SG161222/Realistic_Vision_V5.1_noVAE'), resume_download=True, token=token); \
# snapshot_download(repo_id='guoyww/animatediff-motion-module-v3', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'guoyww/animatediff-motion-module-v3'), resume_download=True, token=token); \
# snapshot_download(repo_id='lllyasviel/control_v11p_sd15_openpose', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'lllyasviel/control_v11p_sd15_openpose'), resume_download=True, token=token); \
# snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'lllyasviel/control_v11f1p_sd15_depth'), resume_download=True, token=token); \
# snapshot_download(repo_id='h94/IP-Adapter', allow_patterns=['models/image_encoder/*', 'ip-adapter_sd15.bin'], local_dir=os.path.join(hf_cache_dir, 'h94/IP-Adapter'), resume_download=True, token=token); \
# "

# Uncomment the following section to install Git LFS if your models might use it
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends git-lfs && \
#     git lfs install && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

COPY main.py ./main.py

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/healthz || exit 1

CMD ["python", "main.py"]
