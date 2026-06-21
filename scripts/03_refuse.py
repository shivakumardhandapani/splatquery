#!/usr/bin/env python
"""Rebuild the map from cached detections - fast (seconds, no SAM2/CLIP rerun).

Use this to re-tune fusion (merge thresholds, voxel size, min points) after a
one-time `01_build_map.py`. It reads detections.pkl, re-runs only the clustering
+ merge, and overwrites the map.

Usage:
    python scripts/03_refuse.py --cache runs/room0/detections.pkl \
        --out runs/room0/map.pkl \
        --set mapping.merge_overlap_threshold=0.25 mapping.cluster_eps=0.12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatquery.config import load_config
from splatquery.pipeline import fuse_detections, load_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default="runs/room0/detections.pkl")
    ap.add_argument("--out", default="runs/room0/map.pkl")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    detections = load_detections(args.cache)
    print(f"[refuse] loaded {len(detections)} cached detections")
    smap = fuse_detections(detections, cfg)
    smap.save(args.out)
    print(f"[done] re-fused into {len(smap)} objects -> {args.out}")


if __name__ == "__main__":
    main()
