import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import replicate 

# Make Streamlit secrets available as env vars for modules that only read os.getenv()
if hasattr(st, "secrets") and "REPLICATE_API_TOKEN" in st.secrets:
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

def get_secret(name):
    v = st.secrets.get(name) if hasattr(st, "secrets") else None
    if not v:
        v = os.getenv(name)
    return v.strip() if isinstance(v, str) else None

REPLICATE_API_TOKEN = get_secret("REPLICATE_API_TOKEN")
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

st.write("replicate token starts with:", (st.secrets.get("REPLICATE_API_TOKEN","")[:4]))


# Optional patches / local modules
import patch_basicsr  # noqa: F401
# patch_basicsr.patch_basicsr()

from dotenv import load_dotenv
load_dotenv()

import requests

# Project modules (keep your existing implementations)
from utils import (
    load_saved_model,
    enhance_image,
    grayscale_except_blue_yellow,
    load_and_preprocess,
)
from upscale import upscale_image  # Replicate/cloud
from local_upscale import upscale_image_local  # Local PyTorch x2
from text2img import generate_image_from_prompt_and_image  # img2img

# Try to load clickable image gallery component
try:
    from streamlit_image_select import image_select
    HAS_IMG_SELECT = True
except Exception:
    HAS_IMG_SELECT = False

# ---------------- Small utilities ----------------
def _save_url_to_file(url: str, local_path: str = "generated_input.jpg") -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path


def _file_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.isfile(path)
    except Exception:
        return False


# ---------------- Global page config & styles ----------------
st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="centered")

# Narrow, centered content for clarity
st.markdown(
    """
    <style>
      .block-container {max-width: 900px !important; padding-top: 1.0rem;}
      .step-hint { margin-top: -0.25rem; opacity: 0.8; }
      .thumb-caption { text-align:center; font-size:0.9rem; opacity:0.85; }
      .stCaption { text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
<style>
/* --- your existing styles here --- */

/* Ensure radio labels use your text color */
[data-testid="stRadio"] [role="radiogroup"] label,
[data-testid="stRadio"] [role="radiogroup"] label p {
  color: var(--text, #111827) !important;  /* falls back to dark text if --text not set */
}

/* Streamlit info/alert box (the blue rectangle) */
[data-testid="stAlert"],
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
  color: var(--text, #111827) !important;
}

/* Optional: make the alert match your panel theme instead of blue */
/*
[data-testid="stAlert"]{
  background: var(--panel, #f8fafc) !important;
  border-left-color: var(--accent, #d62828) !important;
}
/* Image selector background (theme aware) */
[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
    background-color: var(--panel, #ffffff) !important; /* white for light, dark for dark */
    border-radius: 8px;
    transition: background-color 0.3s ease;
}
*/

</style>
""", unsafe_allow_html=True)


# Centered loading overlay (replaces st.spinner)
from contextlib import contextmanager

@contextmanager
def centered_loader(text: str = "Loading..."):
    placeholder = st.empty()
    html = """
        <style>
        ._overlay_ {{
            position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
            background: rgba(0,0,0,0.35); z-index: 9999;
        }}
        ._loader_ {{
            width: 64px; height: 64px; border-radius: 50%;
            border: 6px solid rgba(255,255,255,0.35);
            border-top-color: white; animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        ._text_ {{ color: #fff; margin-top: 14px; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,.5); text-align:center; }}
        </style>
        <div class="_overlay_">
          <div style="display:flex; flex-direction:column; align-items:center;">
            <div class="_loader_"></div>
            <div class="_text_">{text}</div>
          </div>
        </div>
    """.format(text=text)
    placeholder.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


# ---------------- Session state ----------------
if "selected_style" not in st.session_state:
    st.session_state.selected_style = None
if "upscaled_source" not in st.session_state:
    st.session_state.upscaled_source = None
if "upscaled_is_local" not in st.session_state:
    st.session_state.upscaled_is_local = False
if "selected_gallery_image" not in st.session_state:
    st.session_state.selected_gallery_image = None
if "variation_result_url" not in st.session_state:
    st.session_state.variation_result_url = None


def select_style(style_name: str):
    if st.session_state.get("selected_style") != style_name:
        st.session_state.upscaled_source = None
        st.session_state.variation_result_url = None
    st.session_state.selected_style = style_name


# ---- Cache the Keras model per style (prevents reload on every rerun) ----
@st.cache_resource(show_spinner=False)
def get_model_for_style(style_name: str):
    weights = style_weights[style_name]
    return load_saved_model(weights)


# ---------------- Data ----------------
style_paths = {
    "Starry Night": "images/Starry_Night_S.JPG",
    "The Scream": "images/Edvard_Munch_The_Scream.jpg",
    "Great Wave of Kanagawa": "images/The_Great_Wave_off_Kanagawa.jpg",  # NEW
}

style_weights = {
    "The Scream": "models/scream_model_weights_step_105000.weights.h5",
    "Starry Night": "models/model_weights_step_127000.weights.h5",
    "Great Wave of Kanagawa": "models/GreatWave_model_weights_step_28000.weights.h5",  # NEW
}


# Preloaded gallery images
gallery_images = {
    "Budapest Parliament (Sunset)": "images/budapest_parliament_sunset.jpg",
    "Lakeside Sunset": "images/lakeside_sunset.jpg",
    "Mountain Reflection": "images/mountain_reflection.jpg",
    "Budapest Parliament (Night)": "images/budapest_parliament_night.jpeg",
    "Lake Balaton (Aerial)": "images/lake_balaton_aerial.jpg",
}

prompt_presets = {
    "Starry Night": (
        "Repaint this image in the style of Van Gogh’s Starry Night, "
        "preserving the original composition and structure, "
        "with swirling brushstrokes, vibrant blues and yellows, "
        "thick impasto texture, and expressive movement in the sky and water."
    ),
    "The Scream": (
        "Repaint this image in the style of Edvard Munch’s The Scream, "
        "preserving the original composition and perspective, "
        "with bold, wavy brushstrokes, swirling lines that distort the sky and landscape, "
        "intense oranges and deep blues, and a dramatic, emotional atmosphere."
    ),
}

negative_presets = {
    "Starry Night": "cartoon, surreal architecture, distorted proportions, extra objects, abstract shapes",
    "The Scream": "cartoon, extra objects, warped anatomy, abstract shapes unrelated to the scene",
}

DEFAULT_STRENGTH = 0.20  # hidden control


# ---------------- UI ----------------
st.title("Neural Style Transfer")

# Theme toggle (stack it near the title)
mode = st.radio("Theme", ["Light", "Dark"], horizontal=True, index=1, key="theme_mode")

LIGHT = dict(bg="#ffffff", text="#111827", panel="#f8fafc",
             accent="#d62828", accent_hover="#a71d2a")
DARK  = dict(bg="#0b0f19", text="#e5e7eb", panel="#111827",
             accent="#d62828", accent_hover="#a71d2a")

C = DARK if mode == "Dark" else LIGHT

st.markdown(f"""
<style>
:root {{
  --bg: {C['bg']};
  --text: {C['text']};
  --panel: {C['panel']};
  --accent: {C['accent']};
  --accent-hover: {C['accent_hover']};
}}
html, body, .stApp {{ background: var(--bg); color: var(--text); }}
/* Buttons */
div.stButton > button {{
  background-color: var(--accent);
  color: white; font-weight: bold; border-radius: 8px;
  border: 2px solid rgba(0,0,0,.15);
}}
div.stButton > button:hover {{ background-color: var(--accent-hover); }}
/* Cards / panels (optional) */
section.main > div, .st-emotion-cache-18ni7ap, .stSidebar, .stTabs {{ background: var(--panel); }}
</style>
""", unsafe_allow_html=True)


# (1) Choose a style image
st.subheader("1️⃣ Choose a Style Image")
st.caption("Click a style image below to select.")

style_names, style_imgs = [], []
for name, path in style_paths.items():
    if _file_exists(path):
        style_names.append(name)
        style_imgs.append(path)

if style_imgs and HAS_IMG_SELECT:
    chosen_style_img = image_select(
        label="",
        images=style_imgs,
        captions=style_names,
        use_container_width=False,
    )
    if chosen_style_img is not None:
        for n, p in style_paths.items():
            if p == chosen_style_img:
                select_style(n)
                break
else:
    # Fallback radio if component isn't available
    selected_style_name = st.radio("Pick a style:", list(style_paths.keys()), horizontal=True)
    if selected_style_name:
        select_style(selected_style_name)

if st.session_state.selected_style:
    st.success(f"✅ You selected: {st.session_state.selected_style} ✅")

# --- Source choice: default to Upload ---
if "image_source_mode" not in st.session_state:
    st.session_state.image_source_mode = "Upload"  # default

source = st.radio(
    "Choose where to pick your image from:",
    ["Gallery", "Upload"],
    index=0,  # Upload first by default
    horizontal=True,
    key="image_source_mode",
)

st.subheader("2️⃣ Choose from gallery or upload your own")
st.caption("You can upload an image or pick from the gallery — whichever you prefer.")

# --- stacked layout: upload first, then gallery ---
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="uploader")

st.markdown("**Or pick from the gallery:**")

# default gallery selection (first item) so gallery_key always exists
_default_gallery_key = next(iter(gallery_images.keys()))
gallery_key = _default_gallery_key

if HAS_IMG_SELECT:
    gallery_paths = list(gallery_images.values())
    gallery_captions = list(gallery_images.keys())

    chosen_gallery_img = image_select(
        label="",
        images=gallery_paths,
        captions=gallery_captions,
        use_container_width=True
    )

    if chosen_gallery_img is not None:
        # map chosen path back to the caption/key
        for name, path in gallery_images.items():
            if path == chosen_gallery_img:
                gallery_key = name
                break
else:
    # Fallback if image-select isn't available
    gallery_key = st.selectbox("Gallery image", list(gallery_images.keys()), key="gallery_key")

st.divider()

# --- Decide which image to use: uploaded > gallery ---
if uploaded is not None:
    preview_img = Image.open(uploaded).convert("RGB")
    image_source_label = " (uploaded)"
else:
    preview_img = Image.open(gallery_images[gallery_key]).convert("RGB")
    image_source_label = f" (gallery: {gallery_key})"

# Show preview
st.image(preview_img, caption=f"Chosen image {image_source_label}", use_container_width=True)
preview_img.save("temp_input.jpg")



if not st.session_state.selected_style:
    st.warning("Please select a style image above first.")
    st.stop()

# (3) Apply style
st.subheader("3️⃣ Apply Style")

with centered_loader("🖌️ Applying style…"):
    # Load model for selected style (cached per style)
    model = get_model_for_style(st.session_state.selected_style)

    # Enhance and preprocess
    original_img, enhanced_img = enhance_image("temp_input.jpg")
    grayscale_img = grayscale_except_blue_yellow(original_img)

    # Run through model: return is in [0..1], convert to [0..255] uint8
    def run_model(arr_or_path):
        content = load_and_preprocess(arr_or_path, is_array=True)
        out = model(tf.constant(content))[0].numpy()
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    stylized = run_model(original_img)
    stylized2 = run_model(enhanced_img)
    stylized3 = run_model(grayscale_img)

st.image(stylized, caption="Stylized Image", width=420)

st.markdown("### Stylized Variants")
v1, v2, v3 = st.columns(3, gap="large")
with v1:
    st.image(stylized2, caption="Enhanced Stylization", width=300)
with v2:
    st.image(stylized3, caption="Grayscale Filter Stylization", width=300)
with v3:
    st.image(stylized, caption="Original Stylization", width=300)

# Save each variant to disk so we can use them later (e.g., upscaling)
Image.fromarray(stylized2).save("stylized_enhanced.jpg")
Image.fromarray(stylized3).save("stylized_grayscale.jpg")
Image.fromarray(stylized).save("stylized_original.jpg")

# Single-choice selection for which variant to upscale
variant_options = {
    "Enhanced Stylization": "stylized_enhanced.jpg",
    "Grayscale Stylization": "stylized_grayscale.jpg",
    "Original Stylization": "stylized_original.jpg",
}
selected_variant_label = st.radio(
    "Select ONE variant to use for upscaling",
    list(variant_options.keys()),
    index=0,
    horizontal=False,
)
st.session_state["selected_variant_path"] = variant_options[selected_variant_label]

# Show which one is selected
st.info(f"Variant selected: {selected_variant_label}")

stylized_img = Image.fromarray(stylized)
stylized_img.save("stylized_output.jpg")

# (4) Upscale
st.subheader("4️⃣ Upscale")

st.markdown(
    """
**What’s the difference?**  
- **Local x2 (PyTorch):** runs on your machine, keeps data local; simplest to use after setup, fixed **2×** scale, speed depends on your hardware.  
- **Replicate (cloud):** runs on a remote GPU, no local compute required; typically more flexible backends and potentially faster if your local machine is slow.
    """
)

engine = st.radio("Choose upscale engine:", ["Local x2 (PyTorch)", "Replicate (cloud)"], horizontal=True)

if st.button("Upscale selected image"):
    input_path = st.session_state.get("selected_variant_path", "stylized_original.jpg")
    with centered_loader("Upscaling…"):
        if engine.startswith("Local"):
            out_path = upscale_image_local(input_path, ckpt_path="art_srnet_x2.pth", tile=None)
            # Ensure we store a file path for later download; convert PIL.Image to file if needed
            if isinstance(out_path, Image.Image):
                _tmp = "upscaled_output_local.png"
                try:
                    out_path.save(_tmp)
                    out_path = _tmp
                except Exception:
                    # As a fallback, convert via numpy
                    _arr = np.array(out_path.convert("RGB"))
                    Image.fromarray(_arr).save(_tmp)
                    out_path = _tmp
            st.session_state["upscaled_source"] = out_path
            st.session_state["upscaled_is_local"] = True
        else:
            upscaled_url = upscale_image(input_path)
            st.session_state["upscaled_source"] = upscaled_url
            st.session_state["upscaled_is_local"] = False
        # Clear any previous variation result when a new upscale is created
        st.session_state["variation_result_url"] = None

# Always render the latest upscaled result (persist across reruns)
up_src = st.session_state.get("upscaled_source")
if up_src:
    if st.session_state.get("upscaled_is_local", False):
        st.image(up_src, caption="Upscaled (Local)", width=520)
        try:
            with open(up_src, "rb") as f:
                st.download_button("Download upscaled image", data=f, file_name=os.path.basename(up_src))
        except Exception:
            pass
    else:
        st.image(up_src, caption="Upscaled (Replicate)", width=520)
        st.markdown(f"[Download upscaled image]({up_src})")

# (5) Optional: Prompt-guided variation from the upscaled image (img2img)
st.subheader("5️⃣ Optional: Style-guided Variation (img2img)")

src = st.session_state.get("upscaled_source")
if not src:
    st.info("Upscale an image first, then (optionally) generate a style-guided variation.")
else:
    chosen_style = st.session_state.selected_style or "Starry Night"
    default_prompt = prompt_presets.get(chosen_style, "")
    default_negative = negative_presets.get(chosen_style, "")

    with st.expander("Advanced (prompt options)"):
        prompt = st.text_area("Prompt", default_prompt, height=100)
        negative = st.text_input("Negative prompt (optional)", default_negative)
        strength = st.slider("Strength (how much to follow the prompt)", 0.05, 0.75, DEFAULT_STRENGTH, 0.20)
        seed = st.number_input("Seed (optional, -1 = random)", value=-1, step=1)

    if st.button("🎇 Style Variations"):
        with centered_loader("✨ Generating image…"):
            try:
                if st.session_state.get("upscaled_is_local", False):
                    # local upscaled -> pass a local file path
                    init_image_path_or_url = src
                else:
                    # cloud upscaled -> we have a URL already
                    init_image_path_or_url = src

                # If it's a local path (not URL), upload path or pass to API as file
                # (generate_image_from_prompt_and_image should handle both)
                result_url = generate_image_from_prompt_and_image(
                    image_path_or_url=src,
                    prompt=prompt,
                    negative_prompt=negative or None,
                    strength=strength,
                    seed=None if seed == -1 else int(seed),
                )

                st.session_state["variation_result_url"] = result_url
            except Exception as e:
                st.error(f"Img2img failed: {e}")

# Persistently render both the upscaled image and any generated variation
up_src = st.session_state.get("upscaled_source")
var_url = st.session_state.get("variation_result_url")
if up_src:
    # already rendered in step 4; keep it visible above
    pass
if var_url:
    st.image(var_url, caption="Prompt + Upscaled (img2img)", width=520)
    st.markdown(f"[Download generated image]({var_url})")
