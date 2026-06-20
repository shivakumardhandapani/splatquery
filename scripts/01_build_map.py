#!/usr/bin/env python
"""Build the 3D semantic map from a posed RGB-D scene and save it.

Usage:
    python scripts/01_build_map.py --config config/default.yaml \
        --out runs/room0/map.pkl \
        --set dataset.root=data/replica/room_0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatquery.config import load_config
from splatquery.pipeline import build_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="runs/scene/map.pkl")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[],
                    help="dotted overrides, e.g. dataset.root=/path agent.backend=local")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    smap = build_map(cfg)
    smap.save(args.out)
    print(f"[done] saved map with {len(smap)} objects -> {args.out}")


if __name__ == "__main__":
    main()
