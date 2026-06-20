"""CLIP encoder: turns masked image regions and text queries into a shared
embedding space. Region embeddings live on the 3D map; text embeddings come
from user queries. Cosine similarity between them = open-vocabulary retrieval.
"""

from __future__ import annotations

import numpy as np


class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-16",
                 pretrained: str = "laion2b_s34b_b88k",
                 device: str = "cuda", crop_padding: float = 0.15):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.crop_padding = crop_padding
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    # --- image side ---------------------------------------------------------
    def _crop(self, rgb: np.ndarray, bbox, seg) -> np.ndarray:
        from PIL import Image
        h, w = rgb.shape[:2]
        x, y, bw, bh = bbox
        px, py = int(bw * self.crop_padding), int(bh * self.crop_padding)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
        # Soft-mask the background so CLIP focuses on the object, not its surroundings.
        region = rgb.copy()
        region[~seg] = (region[~seg] * 0.35).astype(region.dtype)
        return np.asarray(Image.fromarray(region[y0:y1, x0:x1]))

    def encode_masks(self, rgb: np.ndarray, masks) -> np.ndarray:
        """Return (N, D) L2-normalized embeddings, one per mask."""
        if not masks:
            return np.zeros((0, self.model.visual.output_dim), dtype=np.float32)
        from PIL import Image
        batch = []
        for m in masks:
            crop = self._crop(rgb, m.bbox, m.seg)
            batch.append(self.preprocess(Image.fromarray(crop)))
        x = self.torch.stack(batch).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    # --- text side ----------------------------------------------------------
    def encode_text(self, phrases: list[str]) -> np.ndarray:
        """Return (N, D) L2-normalized text embeddings."""
        tokens = self.tokenizer(phrases).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)
