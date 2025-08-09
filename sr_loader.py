# sr_loader.py
# Loads your ArtSRx2 model and fills it with a training checkpoint dict.

import torch
import torch.nn as nn

# ----- Model defs must match training EXACTLY -----
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.body(x)

class ArtSRx2(nn.Module):
    """Learned ×2 upscaler (PixelShuffle)."""
    def __init__(self, width=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, width, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResBlock(width) for _ in range(n_blocks)],
            nn.Conv2d(width, width, 3, 1, 1)
        )
        self.up = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Conv2d(width, 3, 3, 1, 1)

    def forward(self, x):
        f = self.head(x)
        f = self.body(f) + f
        f = self.up(f)
        out = self.tail(f)
        return out
# ---------------------------------------------------

def get_device(pref=None):
    if pref in ("cuda", "cpu"):
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_model(device=None, width=64, n_blocks=8):
    device = get_device(device)
    model = ArtSRx2(width=width, n_blocks=n_blocks).to(device)
    model.eval()
    return model, device

def _strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint_into_model(model, ckpt_path, device=None, strict=True):
    """
    Loads your training checkpoint dict:
      {'epoch', 'state_dict', 'optimizer', 'loss'}
    Fills 'model' with state_dict and returns (model, meta).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = _strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print("[sr_loader] Missing keys:", missing)
    if unexpected:
        print("[sr_loader] Unexpected keys:", unexpected)

    if device:
        model.to(device)
    model.eval()
    return model, ckpt
