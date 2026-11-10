"""LaMa (big-lama, TorchScript) inpainting for removing the original text."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "big-lama.pt"


class Lama:
    def __init__(self, model_path: Path = DEFAULT_MODEL, device: str | None = None):
        import torch

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[inpaint] downloading LaMa weights -> {model_path}")
            urllib.request.urlretrieve(LAMA_URL, model_path)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()

    def __call__(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint `mask > 0` pixels of a BGR crop. Returns a BGR crop of the same size;
        pixels outside the mask are returned unchanged."""
        torch = self.torch
        h, w = img_bgr.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        m = np.pad((mask > 0).astype(np.float32), ((0, ph), (0, pw)), mode="symmetric")

        t_img = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        t_mask = torch.from_numpy(m)[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(t_img, t_mask)
        out = out[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()[:h, :w]
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        keep = mask[..., None] == 0
        return np.where(keep, img_bgr, out)
