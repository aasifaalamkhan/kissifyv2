import os
import sys
from huggingface_hub import snapshot_download, login
import requests  # To handle errors in a generic way

# --- Configuration ---
HF_HOME_DIR = os.getenv('HF_HOME', '/app/hf_cache')  # Ensure this matches Dockerfile and main.py
MODELS_TO_DOWNLOAD = [
    # (repo_id, allow_patterns)
    ('SG161222/Realistic_Vision_V5.1_noVAE', ['*.safetensors', '*.json']),
    ('guoyww/animatediff-motion-adapter-v1-5-2', ['*.safetensors', '*.json']),
    ('lllyasviel/control_v11p_sd15_openpose', ['*.safetensors', '*.json']),
    ('lllyasviel/control_v11f1p_sd15_depth', ['*.safetensors', '*.json']),
    ('h94/IP-Adapter', ['models/*', 'ip-adapter_sd15.bin']),
    ('openai/clip-vit-large-patch14', None),  # No specific patterns needed, download all
    ('lllyasviel/ControlNet', None),  # No specific patterns needed, download all
]

def is_model_cached(repo_id, patterns):
    """
    Check if the model files for the given repo_id are already cached in HF_HOME_DIR.
    """
    model_dir = os.path.join(HF_HOME_DIR, repo_id.replace("/", "_"))  # HuggingFace caches by repo_id, replace "/" with "_"
    
    if not os.path.exists(model_dir):
        return False  # Directory doesn't exist, model is not cached

    if patterns:  # If specific patterns were provided, check if the required files exist
        for pattern in patterns:
            matching_files = [f for f in os.listdir(model_dir) if f.endswith(pattern)]
            if not matching_files:
                return False  # No matching files found for the pattern
    else:
        # If no patterns, just check if there are any files in the directory
        if not os.listdir(model_dir):
            return False  # Directory is empty, no model files found

    return True  # The model is cached

def download_huggingface_models():
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    # Login with token if available
    if token:
        print('Logging in to Hugging Face Hub...', flush=True)
        try:
            login(token=token, add_to_git_credential=False)
            print('Successfully logged in to Hugging Face Hub.', flush=True)
        except requests.exceptions.RequestException as e:  # Catch network and HTTP errors
            print(f'ERROR: Failed to log in to Hugging Face Hub: {e}', flush=True)
            sys.exit(1)  # Exit to fail the build if login fails
    else:
        print('WARNING: HUGGING_FACE_HUB_TOKEN not set as build arg. Proceeding without explicit login. Gated models may fail.', flush=True)

    print(f'Using HF_HOME: {HF_HOME_DIR}', flush=True)
    os.makedirs(HF_HOME_DIR, exist_ok=True)  # Ensure cache directory exists

    all_downloads_successful = True
    for repo_id, patterns in MODELS_TO_DOWNLOAD:
        # Check if the model is already cached
        if is_model_cached(repo_id, patterns):
            print(f'Model {repo_id} is already cached. Skipping download.', flush=True)
            continue  # Skip downloading if model is already cached

        print(f'Pre-fetching {repo_id} to {HF_HOME_DIR}...', flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=patterns,
                cache_dir=HF_HOME_DIR,
                resume_download=True,
                token=token  # Pass token explicitly here too
            )
            print(f'Successfully pre-fetched {repo_id}.', flush=True)
        except requests.exceptions.RequestException as e:  # Catch download errors
            print(f'ERROR: Failed to download {repo_id}. This might be due to network issues: {e}', flush=True)
            all_downloads_successful = False
        except Exception as e:
            print(f'ERROR: An unexpected error occurred while downloading {repo_id}: {e}', flush=True)
            all_downloads_successful = False

    if all_downloads_successful:
        print('All model pre-fetching attempts complete successfully.', flush=True)
    else:
        print('WARNING: Some model pre-fetching failed. Check the logs above for details. This build may not function correctly.', flush=True)
        sys.exit(1)  # Force build failure if any model failed to download

if __name__ == '__main__':
    download_huggingface_models()
