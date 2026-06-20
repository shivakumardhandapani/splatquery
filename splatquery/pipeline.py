"""Map-building pipeline: posed RGB-D frames -> SemanticMap.

Querying is separate (see scripts/02_query.py) so you build the map once and
ask it many questions cheaply.
"""

from __future__ import annotations

from tqdm import tqdm

from .data.dataset import build_dataset
from .mapping.lifting import lift_frame, MonoDepthEstimator
from .mapping.semantic_map import SemanticMap
from .perception.encoder import CLIPEncoder
from .perception.segmenter import SAM2Segmenter


def build_map(cfg) -> SemanticMap:
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

    print(f"[build] {len(detections)} detections from {len(dataset)} frames")
    smap = SemanticMap.build(
        detections,
        voxel_size=cfg.mapping.voxel_size,
        cluster_eps=cfg.mapping.cluster_eps,
        cluster_min_points=cfg.mapping.cluster_min_points,
        merge_sim_threshold=cfg.mapping.merge_sim_threshold)
    print(f"[build] fused into {len(smap)} object nodes")
    return smap
