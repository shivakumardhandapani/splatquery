"""Open-vocabulary 2D segmentation with SAM 2.

SAM 2 proposes class-agnostic masks for everything in a frame; we attach
language meaning later via CLIP. This keeps the system open-vocabulary -
there is no fixed label set anywhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mask2D:
    seg: np.ndarray      # (H, W) bool
    area: int
    bbox: tuple[int, int, int, int]  # x, y, w, h


class SAM2Segmenter:
    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cuda",
                 min_mask_area: int = 900, max_masks: int = 40):
        # Imported lazily so the rest of the package imports without SAM2 installed.
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2 = build_sam2(model_cfg, checkpoint, device=device,
                          apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.92,
            min_mask_region_area=min_mask_area,
        )
        self.min_mask_area = min_mask_area
        self.max_masks = max_masks

    def segment(self, rgb: np.ndarray) -> list[Mask2D]:
        raw = self.generator.generate(rgb)  # list of dicts
        masks: list[Mask2D] = []
        for m in raw:
            if m["area"] < self.min_mask_area:
                continue
            x, y, w, h = (int(v) for v in m["bbox"])
            masks.append(Mask2D(m["segmentation"].astype(bool), int(m["area"]),
                                (x, y, w, h)))
        masks.sort(key=lambda m: m.area, reverse=True)
        return masks[: self.max_masks]
