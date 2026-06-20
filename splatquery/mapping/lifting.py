"""Lift 2D masked regions into 3D.

For each mask we back-project its depth pixels through the camera intrinsics
and pose to get world-space points, then summarize the region as a single
`Detection` (a robust centroid + its CLIP embedding + a point sample for the
bbox). Detections from many frames are fused into objects in semantic_map.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    centroid: np.ndarray     # (3,) world xyz
    points: np.ndarray       # (M, 3) sampled world points (for bbox/viz)
    embedding: np.ndarray    # (D,) L2-normalized CLIP embedding
    frame_index: int


def backproject(depth: np.ndarray, seg: np.ndarray, K: np.ndarray,
                c2w: np.ndarray, max_points: int = 1500) -> np.ndarray:
    """Return world-space points (M, 3) for the True pixels of `seg`."""
    ys, xs = np.nonzero(seg & (depth > 0) & np.isfinite(depth))
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if xs.size > max_points:                       # subsample big masks
        sel = np.random.choice(xs.size, max_points, replace=False)
        xs, ys = xs[sel], ys[sel]
    z = depth[ys, xs].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)  # (M,4) OpenCV cam
    world = (c2w @ cam.T).T[:, :3]
    return world


def _robust_centroid(points: np.ndarray) -> np.ndarray:
    """Median centroid, then drop far outliers and re-average (stable to depth noise)."""
    med = np.median(points, axis=0)
    d = np.linalg.norm(points - med, axis=1)
    keep = d < (np.median(d) + 2.0 * np.std(d) + 1e-6)
    return points[keep].mean(axis=0) if keep.any() else med


def lift_frame(frame, masks, embeddings: np.ndarray,
               max_points: int = 1500) -> list[Detection]:
    """Combine a frame's masks + their CLIP embeddings into 3D Detections."""
    if frame.depth is None:
        raise ValueError(
            "Frame has no depth. Use a depth-bearing dataset (Replica) or set "
            "dataset.depth_source='mono' so lifting can estimate it.")
    detections: list[Detection] = []
    for mask, emb in zip(masks, embeddings):
        pts = backproject(frame.depth, mask.seg, frame.K, frame.c2w, max_points)
        if pts.shape[0] < 10:
            continue
        detections.append(
            Detection(_robust_centroid(pts), pts, emb, frame.index))
    return detections


class MonoDepthEstimator:
    """Optional metric-ish depth for datasets without GT depth (e.g. LERF).

    Uses Depth-Anything-V2 (relative depth). NOTE: monocular depth is up to an
    unknown scale; for navigation goals you must align it to a metric reference
    (e.g. the COLMAP sparse cloud already produced for your scene). This is the
    one place the dataset-first (Replica) path lets you skip extra calibration.
    """

    def __init__(self, device: str = "cuda",
                 model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model_id, device=device)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        out = self.pipe(Image.fromarray(rgb))["predicted_depth"]
        return np.asarray(out, dtype=np.float32)
