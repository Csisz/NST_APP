# text2img.py
import os, replicate
from dotenv import load_dotenv
load_dotenv()

DEFAULT_TXT2IMG = "replicate/van-gogh-flux"
DEFAULT_IMG2IMG = "replicate/van-gogh-flux"   # full version id should come from .env

import os
try:
    import streamlit as st
except Exception:
    st = None

def get_cfg(name: str, default: str | None = None) -> str | None:
    # prefer Streamlit secrets in the cloud, else env/.env locally
    v = None
    if st and hasattr(st, "secrets"):
        v = st.secrets.get(name)
    if not v:
        v = os.getenv(name)
    return v.strip() if isinstance(v, str) else default

# replace your current uses:
MODEL_ID = get_cfg("REPLICATE_TXT2IMG_VERSION", "replicate/van-gogh-flux")
IMG2IMG_MODEL = get_cfg("REPLICATE_IMG2IMG_VERSION", "replicate/van-gogh-flux")

def generate_image_from_prompt_and_image(
    prompt: str,
    image_path_or_url: str,
    negative_prompt: str = "",
    strength: float = 0.3,          # lower default preserves composition
    guidance_scale: float = 7.5,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    model_id: str | None = None,
):
    # client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    client = replicate.Client(api_token=get_cfg("REPLICATE_API_TOKEN"))
    image_input = open(image_path_or_url, "rb") if os.path.exists(image_path_or_url) else image_path_or_url

    inputs = {
        "image": image_input,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_strength": float(strength),   # 0.35–0.5 preserves content
        "guidance_scale": guidance_scale,     # Flux likes ~2–3.5
        "model": "schnell",                   # or "dev" for slower, finer detail
    }

    # Only set custom sizing when BOTH are provided
    if width and height:
        inputs["aspect_ratio"] = "custom"
        inputs["width"] = int(width)
        inputs["height"] = int(height)

    # Don’t send nulls to Replicate (prevents 422s like aspect_ratio: null)
    inputs = {k: v for k, v in inputs.items() if v is not None}
    
    if seed is not None:
        inputs["seed"] = seed

    mid = model_id or IMG2IMG_MODEL
    try:
        # return URLs instead of FileOutput objects
        # out = client.run(mid, input=inputs, use_file_output=False)
        out = client.run(mid, input=inputs, use_file_output=False)
    except replicate.exceptions.ReplicateError as e:
        if "404" in str(e):
            out = client.run(DEFAULT_IMG2IMG, input=inputs, use_file_output=False)
        else:
            raise
    return out[0] if isinstance(out, list) else out