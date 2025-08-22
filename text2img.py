# text2img.py
import os, replicate
from dotenv import load_dotenv
load_dotenv()

# DEFAULT_TXT2IMG = "replicate/van-gogh-flux"
# DEFAULT_IMG2IMG = "replicate/van-gogh-flux"   # full version id should come from .env
DEFAULT_TXT2IMG = "black-forest-labs/flux-kontext-pro:1d201198f8604c46a30829f17fe80fe6e914eaecba01ff62c5aa16a18f3d4b85"
DEFAULT_IMG2IMG = "black-forest-labs/flux-kontext-pro:1d201198f8604c46a30829f17fe80fe6e914eaecba01ff62c5aa16a18f3d4b85"

import os
try:
    import streamlit as st
except Exception:
    st = None

def get_cfg(name: str, default: str | None = None) -> str | None:
    v = None
    if st and hasattr(st, "secrets"):
        v = st.secrets.get(name)
    if not v:
        v = os.getenv(name)
    return v.strip() if isinstance(v, str) else default

MODEL_ID = get_cfg(
    "REPLICATE_TXT2IMG_VERSION",
    "black-forest-labs/flux-kontext-pro:1d201198f8604c46a30829f17fe80fe6e914eaecba01ff62c5aa16a18f3d4b85"
)

IMG2IMG_MODEL = get_cfg(
    "REPLICATE_IMG2IMG_VERSION",
    "black-forest-labs/flux-kontext-pro:1d201198f8604c46a30829f17fe80fe6e914eaecba01ff62c5aa16a18f3d4b85"
)


def generate_image_from_prompt_and_image(
    prompt: str,
    image_path_or_url: str,
    negative_prompt: str = "",
    strength: float = 0.5,          
    guidance_scale: float = 8.5,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    model_id: str | None = None,
):
    client = replicate.Client(api_token=get_cfg("REPLICATE_API_TOKEN"))
    image_input = open(image_path_or_url, "rb") if os.path.exists(image_path_or_url) else image_path_or_url

    inputs = {
        "image": image_input,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_strength": float(strength),   # Controls how much of the original image is kept
        "guidance_scale": guidance_scale,     # Controls how strongly the prompt is followed
        "num_inference_steps": 30,            # Controls quality/speed tradeoff
    }

    if width and height:
        inputs["width"] = int(width)
        inputs["height"] = int(height)

    if seed is not None:
        inputs["seed"] = seed

    inputs = {k: v for k, v in inputs.items() if v is not None}

    mid = model_id or IMG2IMG_MODEL
    try:
        out = client.run(mid, input=inputs, use_file_output=False)
    except replicate.exceptions.ReplicateError as e:
        if "404" in str(e):
            out = client.run(DEFAULT_IMG2IMG, input=inputs, use_file_output=False)
        else:
            raise
    return out[0] if isinstance(out, list) else out
