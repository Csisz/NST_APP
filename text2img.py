# text2img.py
import os, replicate
from dotenv import load_dotenv
load_dotenv()

# DEFAULT_TXT2IMG = "replicate/van-gogh-flux"
# DEFAULT_IMG2IMG = "replicate/van-gogh-flux"   # full version id should come from .env
DEFAULT_TXT2IMG = "black-forest-labs/flux-kontext-pro"
DEFAULT_IMG2IMG = "black-forest-labs/flux-kontext-pro"



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

MODEL_ID = get_cfg("REPLICATE_TXT2IMG_VERSION", DEFAULT_TXT2IMG)
IMG2IMG_MODEL = get_cfg("REPLICATE_IMG2IMG_VERSION", DEFAULT_IMG2IMG)

# previouse def
# def generate_image_from_prompt_and_image(
#     prompt: str,
#     image_path_or_url: str,
#     negative_prompt: str = "",
#     strength: float = 0.5,          
#     guidance_scale: float = 8.5,
#     width: int | None = None,
#     height: int | None = None,
#     seed: int | None = None,
#     model_id: str | None = None,
# ):
#     client = replicate.Client(api_token=get_cfg("REPLICATE_API_TOKEN"))
#     image_input = open(image_path_or_url, "rb") if os.path.exists(image_path_or_url) else image_path_or_url

#     inputs = {
#         "image": image_input,
#         "prompt": prompt,
#         "negative_prompt": negative_prompt,
#         "prompt_strength": float(strength),   # Controls how much of the original image is kept
#         "guidance_scale": guidance_scale,     # Controls how strongly the prompt is followed
#         "num_inference_steps": 30,            # Controls quality/speed tradeoff
#     }

#     if width and height:
#         inputs["width"] = int(width)
#         inputs["height"] = int(height)

#     if seed is not None:
#         inputs["seed"] = seed

#     inputs = {k: v for k, v in inputs.items() if v is not None}

#     mid = model_id or IMG2IMG_MODEL
#     try:
#         out = client.run(mid, input=inputs, use_file_output=False)
#     except replicate.exceptions.ReplicateError as e:
#         if "404" in str(e):
#             out = client.run(DEFAULT_IMG2IMG, input=inputs, use_file_output=False)
#         else:
#             raise
#     return out[0] if isinstance(out, list) else out

def generate_image_from_prompt_and_image(
    prompt: str,
    image_path_or_url: str,
    negative_prompt: str = "",
    strength: float = 0.5,          # typical: 0.3–0.8
    guidance_scale: float = 8.5,    # typical: 7–12
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
        "negative_prompt": negative_prompt or None,
        "prompt_strength": float(strength),
        "guidance_scale": guidance_scale,
        "num_inference_steps": 30,
    }
    

    inputs = {k: v for k, v in inputs.items() if v is not None}

    mid = model_id or IMG2IMG_MODEL or DEFAULT_IMG2IMG

    # (Optional) show exactly what we’re sending
    st.warning(f"Replicate model: {mid}")
    st.json({k: ("<file>" if hasattr(v, "read") else v) for k, v in inputs.items()})

    out = client.run(mid, input=inputs)

    print("DEBUG type(out):", type(out), "value:", out)

    # If Replicate gives you a file-like object
    if hasattr(out, "url"):
        return out.url()

    # If it gives you a list of file-like objects
    if isinstance(out, list) and out and hasattr(out[0], "url"):
        return out[0].url()

    # If it just gives you a plain URL string
    if isinstance(out, str):
        return out

    return out


