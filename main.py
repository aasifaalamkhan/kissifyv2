import runpod
import torch
import traceback
import os
import io
import base64
import requests
import tempfile
import numpy as np
import gc # Import garbage collection
import time # Import time module for sleep

from threading import Thread
from flask import Flask

from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from controlnet_aux import OpenposeDetector, MidasDetector
from huggingface_hub import hf_hub_download, HfFolder # For better token handling

print("✅ main.py started: Initializing script execution.", flush=True)

# Determine if debug mode is enabled
RP_DEBUG = os.getenv("RP_DEBUG", "False").lower() == "true"
if RP_DEBUG:
    print("✨ Debug mode is ENABLED.", flush=True)

# Optimize cuDNN for consistent input shapes (like image sizes)
torch.backends.cudnn.benchmark = True

# --- IP-Adapter Helper Class ---
class IPAdapterImageProj(torch.nn.Module):
    """
    A simple linear layer to project image embeddings for the IP-Adapter.
    """
    def __init__(self, state_dict):
        super().__init__()
        # Initialize a linear layer based on the shapes found in the state_dict
        # Ensure dimensions match what the state_dict expects
        try:
            input_dim = state_dict["image_proj.weight"].shape[1] # Corrected key based on typical linear layer state_dict
            output_dim = state_dict["image_proj.weight"].shape[0] # Corrected key
            self.image_proj_model = torch.nn.Linear(input_dim, output_dim)
            self.image_proj_model.load_state_dict({
                "weight": state_dict["image_proj.weight"],
                "bias": state_dict["image_proj.bias"] # Assuming bias is present
            })
            if RP_DEBUG:
                print(f"DEBUG: IPAdapterImageProj initialized with input_dim={input_dim}, output_dim={output_dim}", flush=True)
        except KeyError:
            # Fallback if the state_dict keys are different (e.g., from an older IP-Adapter version)
            print("WARNING: 'image_proj.weight' or 'image_proj.bias' not found directly. Attempting alternative state_dict parsing.", flush=True)
            self.image_proj_model = torch.nn.Linear(
                state_dict["image_proj"].shape[-1], state_dict["ip_adapter"].shape[0]
            )
            self.image_proj_model.load_state_dict(
                {"weight": state_dict["image_proj"], "bias": torch.zeros(state_dict["image_proj"].shape[0])}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IPAdapterImageProj: {e}\n{traceback.format_exc()}")

    def forward(self, image_embeds):
        """
        Forward pass for the image projection model.
        """
        return self.image_proj_model(image_embeds)

# --- Globals for Lazy-Loading Models ---
pipe = None
image_encoder = None
image_proj_model = None
image_processor = None
openpose_detector = None
midas_detector = None

# Set Hugging Face cache directory (important if pre-fetching during build)
os.environ['HF_HOME'] = os.getenv('HF_HOME', '/app/hf_cache')
print(f"Hugging Face cache directory set to: {os.environ['HF_HOME']}", flush=True)

# --- File Upload Utility ---
def upload_to_catbox(filepath: str) -> str:
    """
    Uploads a file to Catbox.moe for temporary hosting.

    Args:
        filepath (str): The path to the file to upload.

    Returns:
        str: The URL of the uploaded file or an error message.
    """
    print(f"📤 Attempting to upload {filepath} to Catbox...", flush=True)
    try:
        with open(filepath, 'rb') as f:
            files = {'fileToUpload': (os.path.basename(filepath), f)}
            data = {'reqtype': 'fileupload', 'userhash': ''} # userhash can be empty
            response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data, timeout=60)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
            if response.text.startswith("https://"):
                print(f"✅ Upload successful. URL: {response.text}", flush=True)
                return response.text
            else:
                print(f"❌ Catbox upload returned unexpected response: {response.text}", flush=True)
                return f"Error uploading: Catbox returned unexpected response: {response.text}"
    except requests.exceptions.Timeout:
        print(f"❌ Catbox upload timed out after 60 seconds.", flush=True)
        return "Error uploading: Catbox upload timed out."
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or HTTP error during Catbox upload: {e}", flush=True)
        print(f"Response content (if any): {response.text if 'response' in locals() else 'N/A'}", flush=True)
        return f"Error uploading: Network or HTTP issue: {e}"
    except Exception as e:
        print(f"❌ Unexpected error uploading to Catbox: {traceback.format_exc()}", flush=True)
        return f"Error uploading: {str(e)}"

# --- Health Check Server ---
def run_healthcheck_server():
    """
    Starts a Flask web server to respond to health check requests.
    This ensures RunPod knows the worker is alive and responsive.
    """
    app = Flask(__name__)

    @app.route('/healthz')
    def health():
        """Responds with 'ok' to health checks."""
        if RP_DEBUG:
            print("💖 Health check received. Responding 'ok'.", flush=True)
        return "ok"

    try:
        # Run the Flask app, accessible from outside the container on port 3000.
        # debug=False and use_reloader=False are important for production deployment.
        print("Starting Flask health check server on 0.0.0.0:3000...", flush=True)
        app.run(host="0.0.0.0", port=3000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Health check server failed to start: {traceback.format_exc()}", flush=True)
        # It's critical for this thread to run, if it fails, the container will eventually be marked unhealthy.

# --- Job Handler: Main Video Generation Logic ---
def generate_video(job: dict) -> dict:
    """
    Main job handler for RunPod. Generates a video based on the input image and parameters.

    Args:
        job (dict): The job payload from RunPod, containing 'input' parameters.

    Returns:
        dict: A dictionary containing the video URL or an error message.
    """
    job_id = job.get('id', 'N/A')
    print(f"📥 Job received. Processing job ID: {job_id}", flush=True)
    global pipe, image_encoder, image_proj_model, image_processor, openpose_detector, midas_detector

    # Lazy-load models on the first job
    if pipe is None:
        print("⏳ Models not loaded. Beginning model loading process...", flush=True)
        # Add a delay to ensure network is fully ready before model downloads
        print("Waiting 5 seconds for network to stabilize...", flush=True)
        time.sleep(5)
        print("Resuming model loading...", flush=True)

        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available! A GPU is absolutely required for this application.")
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}", flush=True)
            print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)

            # Retrieve Hugging Face token from environment variables or HfFolder
            HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or HfFolder.get_token()
            if HUGGING_FACE_TOKEN:
                print("🔐 Hugging Face token detected.", flush=True)
                # Set token for session
                from huggingface_hub import login
                login(token=HUGGING_FACE_TOKEN, add_to_git_credential=False)
            else:
                print("⚠️ Hugging Face token not found. Downloads for gated models may fail. "
                      "Set HUGGING_FACE_HUB_TOKEN environment variable or login using `huggingface_hub.login()`.", flush=True)

            # Define model IDs
            base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            motion_module_id = "guoyww/animatediff-motion-module-v3"
            ip_adapter_repo_id = "h94/IP-Adapter"
            controlnet_openpose_id = "lllyasviel/control_v11p_sd15_openpose"
            controlnet_depth_id = "lllyasviel/control_v11f1p_sd15_depth"
            controlnet_aux_id = "lllyasviel/ControlNet" # For OpenposeDetector and MidasDetector

            # Load ControlNet models
            print(f"  Loading OpenPose ControlNet from {controlnet_openpose_id}...", flush=True)
            openpose_controlnet = ControlNetModel.from_pretrained(
                controlnet_openpose_id, torch_dtype=torch.float16, use_safetensors=True
            )
            print(f"  Loading Depth ControlNet from {controlnet_depth_id}...", flush=True)
            depth_controlnet = ControlNetModel.from_pretrained(
                controlnet_depth_id, torch_dtype=torch.float16, use_safetensors=True
            )
            print("  ControlNet models loaded.", flush=True)

            # Load ControlNet auxiliary detectors
            print(f"  Loading OpenposeDetector from {controlnet_aux_id}...", flush=True)
            openpose_detector = OpenposeDetector.from_pretrained(
                controlnet_aux_id
            )
            print(f"  Loading MidasDetector from {controlnet_aux_id}...", flush=True)
            midas_detector = MidasDetector.from_pretrained(
                controlnet_aux_id
            )
            print("  ControlNet auxiliary detectors loaded.", flush=True)

            # Load AnimateDiff pipeline
            print(f"  Loading AnimateDiff pipeline from {base_model_id}...", flush=True)
            pipe = AnimateDiffPipeline.from_pretrained(
                base_model_id,
                controlnet=[openpose_controlnet, depth_controlnet],
                torch_dtype=torch.float16,
                use_safetensors=True # Ensure using safetensors if available
            )
            print("  AnimateDiff pipeline loaded. Loading scheduler and motion module...", flush=True)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.load_motion_module(
                motion_module_id, unet_additional_kwargs={"use_inflated_groupnorm": True}
            )
            print("  Scheduler and motion module loaded. Enabling model CPU offload...", flush=True)
            # Offload models to CPU when not in use to save GPU memory
            pipe.enable_model_cpu_offload()
            if RP_DEBUG:
                print("DEBUG: pipe.device (after offload):", pipe.device, flush=True)

            # Load IP-Adapter components
            print(f"  Loading IP-Adapter components from {ip_adapter_repo_id}...", flush=True)
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float16
            ).to("cuda") # Ensure image encoder is on CUDA
            image_processor = CLIPImageProcessor.from_pretrained(
                ip_adapter_repo_id, subfolder="models/image_encoder"
            )
            print("  Downloading IP-Adapter weights...", flush=True)
            ip_adapter_path = hf_hub_download(
                repo_id=ip_adapter_repo_id, filename="ip-adapter_sd15.bin"
            )
            print(f"  Loading IP-Adapter weights from {ip_adapter_path}...", flush=True)
            ip_adapter_weights = torch.load(ip_adapter_path, map_location="cpu")
            # Filter the state_dict to only include keys starting with "image_proj" and "ip_adapter"
            # This is a common practice when loading specific components
            filtered_weights = {k: v for k, v in ip_adapter_weights.items() if k.startswith(("image_proj", "ip_adapter"))}
            image_proj_model = IPAdapterImageProj(filtered_weights).to("cuda") # Ensure projection model is on CUDA
            print("  IP-Adapter components loaded.", flush=True)

            print("✅ All models loaded successfully to GPU.", flush=True)

            # Clear memory after loading
            gc.collect()
            torch.cuda.empty_cache()
            if RP_DEBUG:
                print(f"DEBUG: CUDA memory after initial load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

        except RuntimeError as e:
            error_message = f"❌ CUDA or Model Initialization Critical Error: {e}\n{traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}
        except Exception as e:
            error_message = f"❌ Model loading failed unexpectedly: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

    # --- Parse Job Input ---
    job_input = job.get('input', {})
    base64_image = job_input.get('init_image')
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic')
    if RP_DEBUG:
        print(f"DEBUG: Prompt received: '{prompt}'", flush=True)

    # Parameters with default values and validation
    num_frames = int(job_input.get('num_frames', 16))
    fps = int(job_input.get('fps', 8))

    # Ensure scales are within reasonable bounds (0.0 to 2.0, adjusted from 1.5 for more flexibility)
    ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.7))
    ip_adapter_scale = min(max(ip_adapter_scale, 0.0), 2.0)
    openpose_scale = float(job_input.get('openpose_scale', 1.0))
    openpose_scale = min(max(openpose_scale, 0.0), 2.0)
    depth_scale = float(job_input.get('depth_scale', 0.5))
    depth_scale = min(max(depth_scale, 0.0), 2.0)

    if RP_DEBUG:
        print(f"DEBUG: num_frames={num_frames}, fps={fps}, ip_adapter_scale={ip_adapter_scale}, "
              f"openpose_scale={openpose_scale}, depth_scale={depth_scale}", flush=True)

    if not base64_image:
        print("❌ 'init_image' (base64 encoded) is missing in job input.", flush=True)
        return {"error": "Missing 'init_image' base64 input. Please provide a base64 encoded image."}

    try:
        # Decode base64 image and convert to RGB
        print("🖼️ Decoding base64 image...", flush=True)
        init_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
        print(f"🖼️ Input image dimensions: {init_image.size[0]}x{init_image.size[1]}", flush=True)
    except Exception as e:
        error_message = f"❌ Failed to decode or open 'init_image': {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # --- Image Preprocessing ---
    print("🔍 Preprocessing input image for model inference...", flush=True)
    try:
        # Process image for CLIP (IP-Adapter)
        print("  Processing image for CLIP (IP-Adapter)...", flush=True)
        processed_image = image_processor(images=init_image, return_tensors="pt").pixel_values.to("cuda", dtype=torch.float16)
        clip_features = image_encoder(processed_image).image_embeds
        image_embeds = image_proj_model(clip_features)
        if RP_DEBUG:
            print(f"DEBUG: clip_features shape: {clip_features.shape}", flush=True)
            print(f"DEBUG: image_embeds shape: {image_embeds.shape}", flush=True)

        # Generate ControlNet conditioning images
        print("  Generating OpenPose conditioning image...", flush=True)
        openpose_image = openpose_detector(init_image)
        if RP_DEBUG:
            print(f"DEBUG: OpenPose image generated. Size: {openpose_image.size}", flush=True)
        print("  Generating Depth conditioning image...", flush=True)
        depth_image = midas_detector(init_image)
        if RP_DEBUG:
            print(f"DEBUG: Depth image generated. Size: {depth_image.size}", flush=True)

        control_images = [openpose_image, depth_image]

        cross_attention_kwargs = {"ip_adapter_image_embeds": image_embeds, "scale": ip_adapter_scale}
        print("🔍 Image preprocessing complete.", flush=True)

    except Exception as e:
        error_message = f"❌ Image preprocessing failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # Clear memory before inference
    gc.collect()
    torch.cuda.empty_cache()
    if RP_DEBUG:
        print(f"DEBUG: CUDA memory before inference: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    # --- Video Inference ---
    print("✨ Starting video generation inference...", flush=True)
    try:
        output = pipe(
            prompt=prompt,
            negative_prompt="ugly, distorted, low quality, cropped, blurry, bad anatomy, bad quality, long_neck, long_body, text, watermark, signature", # Added more negative prompts
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=20,
            image=control_images, # ControlNet conditioning images
            controlnet_conditioning_scale=[openpose_scale, depth_scale], # Scales for each ControlNet
            cross_attention_kwargs=cross_attention_kwargs
        )
        frames = output.frames[0] # Assuming we want the first (and likely only) generated video
        print("✅ Video inference completed.", flush=True)

    except torch.cuda.OutOfMemoryError:
        error_message = "❌ CUDA Out Of Memory error during inference. Try reducing num_frames or image resolution. Current memory usage might be too high."
        print(error_message, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if RP_DEBUG:
            print(f"DEBUG: CUDA memory after OOM attempt to clear: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
        return {"error": error_message}
    except Exception as e:
        error_message = f"❌ Video inference failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # Clear memory after inference
    del output
    # frames is a list of PIL Images, which can be large. Explicitly clear if possible.
    # Note: `del frames` only removes the reference, GC will clean up later.
    # If `frames` is a large list, consider processing it in chunks or directly writing to video.
    # For now, relying on Python's GC and empty_cache.
    gc.collect()
    torch.cuda.empty_cache()
    if RP_DEBUG:
        print(f"DEBUG: CUDA memory after inference: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    # --- Export and Upload Video ---
    print("📼 Exporting video and preparing for upload...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify_video.mp4")
        try:
            export_to_video(frames, video_path, fps=fps)
            print(f"📼 Video exported to {video_path} with {fps} FPS.", flush=True)
        except Exception as e:
            error_message = f"❌ Failed to export video: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

        print("🚀 Uploading generated video to Catbox...", flush=True)
        video_url = upload_to_catbox(filepath=video_path)
        if "Error" in video_url:
            print(f"❌ Video upload failed: {video_url}", flush=True)
            return {"error": video_url}

        print(f"✅ Video generation complete and uploaded. URL: {video_url}", flush=True)
        return {"output": {"video_url": video_url}}

# --- Entry Point for RunPod Worker ---
if __name__ == "__main__":
    # Start the health check server in a separate daemon thread.
    print("Starting health check server thread...", flush=True)
    try:
        health_thread = Thread(target=run_healthcheck_server, daemon=True)
        health_thread.start()
        print("Health check server thread started.", flush=True)
    except Exception as e:
        print(f"❌ Failed to start health check server thread: {traceback.format_exc()}", flush=True)
        # If health check fails to start, the worker will likely be marked unhealthy quickly.
        exit(1) # Exit immediately if essential service cannot start

    try:
        print("🚀 RunPod worker is ready to receive jobs...", flush=True)
        runpod.serverless.start({"handler": generate_video})
    except Exception as e:
        print(f"❌ RunPod serverless failed to start: {traceback.format_exc()}", flush=True)
        # It's good practice to let the container exit if the main service fails to initialize.
        # An uncaught exception here will cause the container to exit with a non-zero code.
        exit(1) # Ensure the container exits with an error code

    # --- Local Test Mode (Optional) ---
    # Uncomment the following block for local testing without RunPod.
    # Requires a base64 encoded image string for `base64_test_image`.
    # Make sure to comment out `runpod.serverless.start` if testing locally.

    # print("\n--- Running Local Test Mode (if uncommented) ---", flush=True)
    # try:
    #     # IMPORTANT: Replace with a real base64 image string for testing!
    #     # To generate a base64 string from a file:
    #     # import base64
    #     # with open("path/to/your/image.jpg", "rb") as image_file:
    #     #     base64_test_image = base64.b64encode(image_file.read()).decode('utf-8')
    #
    #     # base64_test_image = "YOUR_BASE64_IMAGE_STRING_HERE" # <<< IMPORTANT: REPLACE THIS
    #
    #     # if base64_test_image == "YOUR_BASE64_IMAGE_STRING_HERE":
    #     #     print("⚠️ WARNING: Please replace 'YOUR_BASE64_IMAGE_STRING_HERE' with a valid base64 image for local testing.")
    #     # else:
    #     #     fake_job = {
    #     #         "id": "local-test-job",
    #     #         "input": {
    #     #             "init_image": base64_test_image,
    #     #             "prompt": "a couple kissing under the moonlight, cinematic, romantic",
    #     #             "num_frames": 16,
    #     #             "fps": 8,
    #     #             "ip_adapter_scale": 0.8,
    #     #             "openpose_scale": 1.2,
    #     #             "depth_scale": 0.6
    #     #         }
    #     #     }
    #     #     print("\n--- Starting local test job ---", flush=True)
    #     #     result = generate_video(fake_job) # Call synchronously for local testing
    #     #     print("Local test result:", result, flush=True)
    #
    # except Exception as e:
    #     print(f"❌ Local test failed: {traceback.format_exc()}", flush=True)
