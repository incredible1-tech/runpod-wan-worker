"""
RunPod Serverless Handler for Wan2.1-T2V-1.3B Text-to-Video Generation.

Diffusers-native model, run directly via WanPipeline. The UMT5 text
encoder (~23GB, shared across every Wan model size) is kept on CPU and
only the small transformer + VAE live on GPU - this is exactly what
Wan's own CLI calls "--t5_cpu", and it's what gets the 1.3B model down
to its documented ~8GB VRAM footprint instead of ~14GB+.

Input format:
{
    "input": {
        "prompt": "A cat walks on the grass, realistic",
        "negative_prompt": "...",   # optional, sensible default used if omitted
        "width": 832,                # optional, default 832
        "height": 480,                # optional, default 480
        "num_frames": 81,            # optional, default 81 (~5s at 16fps)
        "fps": 16,                   # optional, default 16
        "guidance_scale": 5.0,       # optional, default 5.0
        "seed": null                  # optional, random if not set
    }
}

Output format:
{
    "video": "data:video/mp4;base64,...",
    "duration_seconds": 5.06,
    "resolution": "832x480",
    "seed": 12345,
    "generation_time_seconds": 42.1
}
"""

import base64
import os
import tempfile
import time
from typing import Optional

import runpod
import torch

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

DEFAULT_NEGATIVE_PROMPT = (
    "anime, cartoon, illustration, 2D, cel-shaded, manga, CGI, 3D render, "
    "video game, low-poly, painting, drawing, sketch, watermark, text, "
    "subtitle, logo, bright tones, overexposed, static, blurred details, "
    "worst quality, low quality, jpeg artifacts, ugly, deformed, "
    "disfigured, extra limbs, poorly drawn hands, poorly drawn face, "
    "mutated, still image, out of frame, duplicate"
)

_PIPELINE = None


def load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    print("Loading Wan2.1-T2V-1.3B pipeline...")
    start = time.time()

    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import UMT5EncoderModel

    # Text encoder stays on CPU - it's the single biggest chunk of this
    # model by far and isn't needed on GPU past the encode_prompt call.
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae.to("cuda")

    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
    pipe.transformer.to("cuda")

    _PIPELINE = pipe
    print(f"Pipeline loaded in {time.time() - start:.1f}s")
    return pipe


def generate_video(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    fps: int = 16,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
) -> dict:
    from diffusers.utils import export_to_video

    pipe = load_pipeline()

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating video: {prompt[:60]}...")
    start = time.time()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    generation_time = time.time() - start
    print(f"Generation completed in {generation_time:.1f}s")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.mp4")
        export_to_video(output, output_path, fps=fps)
        with open(output_path, "rb") as f:
            video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    duration = num_frames / fps

    return {
        "video": f"data:video/mp4;base64,{video_base64}",
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
        "fps": fps,
        "seed": seed,
        "generation_time_seconds": round(generation_time, 2),
    }


def handler(event: dict) -> dict:
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        result = generate_video(
            prompt=prompt,
            negative_prompt=input_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            width=input_data.get("width", 832),
            height=input_data.get("height", 480),
            num_frames=input_data.get("num_frames", 81),
            fps=input_data.get("fps", 16),
            guidance_scale=input_data.get("guidance_scale", 5.0),
            seed=input_data.get("seed"),
        )
        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
