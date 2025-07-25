# download_models.py
import os
import sys
from huggingface_hub import snapshot_download, login, HfHubInvalidToken, HfHubDownloadError

# --- Configuration ---
HF_HOME_DIR = os.getenv('HF_HOME', '/app/hf_cache') # Ensure this matches Dockerfile and main.py
MODELS_TO_DOWNLOAD = [
    # (repo_id, allow_patterns)
    ('SG161222/Realistic_Vision_V5.1_noVAE', ['*.safetensors', '*.json']),
    ('guoyww/animatediff-motion-adapter-v1-5-2', ['*.safetensors', '*.json']),
    ('lllyasviel/control_v11p_sd15_openpose', ['*.safetensors', '*.json']),
    ('lllyasviel/control_v11f1p_sd15_depth', ['*.safetensors', '*.json']),
    ('h94/IP-Adapter', ['models/*', 'ip-adapter_sd15.bin']),
    ('openai/clip-vit-large-patch14', None), # No specific patterns needed, download all
    ('lllyasviel/ControlNet', None), # No specific patterns needed, download all
]

def download_huggingface_models():
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        print('Logging in to Hugging Face Hub...', flush=True)
        try:
            login(token=token, add_to_git_credential=False)
            print('Successfully logged in to Hugging Face Hub.', flush=True)
        except HfHubInvalidToken:
            print('ERROR: HUGGING_FACE_HUB_TOKEN is invalid. Please check your token.', flush=True)
            sys.exit(1) # Exit to fail the build if token is bad
        except Exception as e:
            print(f'ERROR: Failed to log in to Hugging Face Hub: {e}', flush=True)
            sys.exit(1)
    else:
        print('WARNING: HUGGING_FACE_HUB_TOKEN not set as build arg. Proceeding without explicit login. Gated models may fail.', flush=True)

    print(f'Using HF_HOME: {HF_HOME_DIR}', flush=True)
    os.makedirs(HF_HOME_DIR, exist_ok=True) # Ensure cache directory exists

    all_downloads_successful = True
    for repo_id, patterns in MODELS_TO_DOWNLOAD:
        print(f'Pre-fetching {repo_id} to {HF_HOME_DIR}...', flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=patterns,
                cache_dir=HF_HOME_DIR,
                resume_download=True,
                token=token # Pass token explicitly here too
            )
            print(f'Successfully pre-fetched {repo_id}.', flush=True)
        except HfHubDownloadError as e:
            print(f'ERROR: Failed to download {repo_id}. This might be due to a missing/invalid token or network issue: {e}', flush=True)
            all_downloads_successful = False
            # Don't sys.exit here, try to download other models. Build will still fail if all_downloads_successful is False.
        except Exception as e:
            print(f'ERROR: An unexpected error occurred while downloading {repo_id}: {e}', flush=True)
            all_downloads_successful = False

    if all_downloads_successful:
        print('All model pre-fetching attempts complete successfully.', flush=True)
    else:
        print('WARNING: Some model pre-fetching failed. Check the logs above for details. This build may not function correctly.', flush=True)
        # You might want to sys.exit(1) here to force a build failure if any download fails,
        # but for now, we'll let it complete to give more verbose logs.
        sys.exit(1) # Force build failure if any model failed to download

if __name__ == '__main__':
    download_huggingface_models()