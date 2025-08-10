# Local x2 super-resolution using your ArtSRx2 checkpoint.

import os
import math
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from sr_loader import build_model, load_checkpoint_into_model, get_device

def _to_tensor(img: Image.Image) -> torch.Tensor:
    # Training used torchvision ToTensor() => float32 in [0,1], no normalization.
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t

def _to_image(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return Image.fromarray((t * 255.0).astype(np.uint8))

@torch.no_grad()
def _forward(model: torch.nn.Module, x: torch.Tensor, use_half: bool) -> torch.Tensor:
    if use_half and x.device.type == "cuda":
        model = model.half()
        x = x.half()
    y = model(x)
    return y

def _overlap_tiles(h: int, w: int, tile: int, overlap: int) -> Tuple[list, int, int]:
    ys = list(range(0, h, tile - overlap))
    xs = list(range(0, w, tile - overlap))
    if ys and ys[-1] + tile > h: ys[-1] = max(0, h - tile)
    if xs and xs[-1] + tile > w: xs[-1] = max(0, w - tile)
    return [(y, x) for y in ys for x in xs], len(ys), len(xs)

@torch.no_grad()
def _forward_tiled(model, x, use_half: bool, tile: int = 256, overlap: int = 16) -> torch.Tensor:
    """
    Simple tile forward with blending to reduce seams.
    Input x: (1,3,H,W) -> Output: (1,3,2H,2W)
    """
    b, c, h, w = x.shape
    tiles, ny, nx = _overlap_tiles(h, w, tile, overlap)
    scale = 2  # ArtSRx2 is x2

    out = torch.zeros((1, 3, h * scale, w * scale), dtype=torch.float32, device=x.device)
    weight = torch.zeros_like(out)

    for (y, x0) in tiles:
        y1 = min(y + tile, h)
        x1 = min(x0 + tile, w)
        patch = x[:, :, y:y1, x0:x1]

        # soft alpha to blend overlaps
        ph, pw = patch.shape[-2:]
        alpha_y = torch.ones(ph, device=patch.device)
        alpha_x = torch.ones(pw, device=patch.device)
        edge = overlap // 2
        if ph > 2 * edge:
            ramp = torch.linspace(0, 1, steps=edge, device=patch.device)
            alpha_y[:edge] = ramp
            alpha_y[-edge:] = ramp.flip(0)
        if pw > 2 * edge:
            ramp = torch.linspace(0, 1, steps=edge, device=patch.device)
            alpha_x[:edge] = ramp
            alpha_x[-edge:] = ramp.flip(0)
        alpha = alpha_y.view(ph, 1) * alpha_x.view(1, pw)
        alpha = alpha.clamp(0, 1)

        patch_up = _forward(model, patch, use_half)   # (1,3,ph*2,pw*2)
        ay, ax = alpha.shape
        alpha_up = torch.kron(alpha, torch.ones((2, 2), device=alpha.device))  # x2

        oy, ox = y * scale, x0 * scale
        out[:, :, oy:oy + ph * scale, ox:ox + pw * scale] += patch_up * alpha_up
        weight[:, :, oy:oy + ph * scale, ox:ox + pw * scale] += alpha_up

    out = out / weight.clamp_min(1e-6)
    return out

@torch.no_grad()
def upscale_image_local(
    input_path: str,
    ckpt_path: str = "art_srnet_x2.pth",
    output_path: Optional[str] = None,
    use_half: bool = True,
    tile: Optional[int] = None,
    overlap: int = 16,
    device: Optional[str] = None,
) -> str:
    """
    Runs local x2 upscaling using your checkpoint.
    - input_path: path to LR image (any HxW; model outputs 2H x 2W)
    - ckpt_path: training checkpoint dict you saved during training
    - tile: set (e.g. 256) for low-VRAM tiling, or None for whole image
    """
    device = get_device(device)
    model, _ = build_model(device=device)                     # matches training arch
    model, _meta = load_checkpoint_into_model(model, ckpt_path, device=device)

    img = Image.open(input_path).convert("RGB")
    x = _to_tensor(img).to(device)

    if tile is None:
        y = _forward(model, x, use_half=use_half)
    else:
        y = _forward_tiled(model, x, use_half=use_half, tile=tile, overlap=overlap)

    out_img = _to_image(y)
    if output_path is None:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_x2_local.jpg"
    out_img.save(output_path)
    return output_path
