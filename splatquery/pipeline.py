"""Map-building pipeline: posed RGB-D frames -> SemanticMap.

Split into two stages so the expensive half runs once:
    extract_detections()  -- SAM2 + CLIP + 3D lifting  (slow: minutes)
    fuse_detections()     -- cluster + merge into objects (fast: seconds)

Caching the detections to disk lets you re-tune fusion (merge thresholds, voxel
size) and rebuild the map in seconds instead of re-running SAM2 every time.

Querying is separate (see scripts/02_query.py) so you build once, ask many times.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from tqdm import tqdm

from .data.dataset import build_dataset
from .mapping.lifting import lift_frame, MonoDepthEstimator
from .mapping.semantic_map import SemanticMap
from .perception.encoder import CLIPEncoder
from .perception.segmenter import SAM2Segmenter


def extract_detections(cfg) -> list:
    """The expensive stage: per-frame SAM2 masks -> CLIP embeddings -> 3D detections."""
    device = cfg.device
    dataset = build_dataset(cfg)

    segmenter = SAM2Segmenter(
        cfg.perception.sam2_checkpoint, cfg.perception.sam2_config, device,
        cfg.perception.min_mask_area, cfg.perception.max_masks_per_frame)
    encoder = CLIPEncoder(
        cfg.perception.clip_model, cfg.perception.clip_pretrained, device,
        cfg.perception.crop_padding)

    mono = None
    if cfg.dataset.get("depth_source", "gt") == "mono":
        mono = MonoDepthEstimator(device)

    detections = []
    for frame in tqdm(dataset, total=len(dataset), desc="frames"):
        if frame.depth is None and mono is not None:
            frame.depth = mono(frame.rgb)   # NOTE: relative scale - see lifting.py
        masks = segmenter.segment(frame.rgb)
        if not masks:
            continue
        embeddings = encoder.encode_masks(frame.rgb, masks)
        detections.extend(lift_frame(frame, masks, embeddings))

    print(f"[extract] {len(detections)} detections from {len(dataset)} frames")
    return detections


def fuse_detections(detections, cfg) -> SemanticMap:
    """The cheap stage: cluster + merge detections into object nodes."""
    smap = SemanticMap.build(
        detections,
        voxel_size=cfg.mapping.voxel_size,
        cluster_eps=cfg.mapping.cluster_eps,
        cluster_min_points=cfg.mapping.cluster_min_points,
        merge_sim_threshold=cfg.mapping.merge_sim_threshold,
        merge_overlap_threshold=cfg.mapping.get("merge_overlap_threshold", 0.2))
    print(f"[fuse] fused into {len(smap)} object nodes")
    return smap


def save_detections(detections, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(detections, fh)
    print(f"[cache] saved {len(detections)} detections -> {path}")


def load_detections(path: str | Path) -> list:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_map(cfg) -> SemanticMap:
    """Full pipeline: extract then fuse (no caching). Kept for convenience."""
    return fuse_detections(extract_detections(cfg), cfg)
