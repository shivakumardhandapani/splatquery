#!/usr/bin/env python
"""Verify a language query by projecting its matched 3D object back onto the
real RGB frames it appears in, and saving annotated PNGs. No OpenGL/window
needed - works headless on WSL.

The projection is the exact inverse of the lifting step that built the map, so
it is self-consistent with however the map was constructed.

Usage:
    python viz/verify_query.py --map runs/room0/map.pkl --ask "a sofa" \
        --set dataset.root=data/Replica/room0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatquery.config import load_config
from splatquery.data.dataset import build_dataset
from splatquery.mapping.semantic_map import SemanticMap
from splatquery.perception.encoder import CLIPEncoder


def project(points, K, c2w):
    """World points -> pixel coords (u,v), validity mask, and depth z."""
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cam = (w2c @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe_z + K[1, 2]
    return np.stack([u, v], axis=1), z > 1e-3, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--map", required=True)
    ap.add_argument("--ask", required=True)
    ap.add_argument("--out", default="runs/room0/verify")
    ap.add_argument("--topframes", type=int, default=3)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    smap = SemanticMap.load(args.map)
    enc = CLIPEncoder(cfg.perception.clip_model, cfg.perception.clip_pretrained,
                      cfg.device)

    hits = smap.query(enc.encode_text([args.ask]), top_k=1)
    if not hits:
        print("no match in map")
        return
    node = hits[0].node
    print(f'matched node {node.node_id} (score {hits[0].score:.3f}) for "{args.ask}"')
    pts = node.points

    # Scan frames; rank by how many of the node's points land in view.
    cache, counts = [], []
    for frame in build_dataset(cfg):
        uv, valid, _ = project(pts, frame.K, frame.c2w)
        h, w = frame.hw
        inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
              (uv[:, 1] >= 0) & (uv[:, 1] < h)
        cache.append((frame.rgb, uv, inb))
        counts.append(int(inb.sum()))

    order = np.argsort(counts)[::-1][: args.topframes]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for rank, idx in enumerate(order):
        if counts[idx] == 0:
            continue
        rgb, uv, inb = cache[idx]
        img = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(img)
        pix = uv[inb]
        for u, v in pix:
            draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 0, 0))
        x0, y0 = pix.min(0)
        x1, y1 = pix.max(0)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=4)
        draw.text((x0 + 4, max(0, y0 - 14)),
                  f'{args.ask}  ({hits[0].score:.2f})', fill=(0, 255, 0))
        safe = args.ask.replace(" ", "_")
        path = outdir / f"{safe}_view{rank}.png"
        img.save(path)
        print(f"  saved {path}  ({counts[idx]} points on object)")
        saved += 1

    if saved:
        print(f"\nOpen the PNGs in {outdir} to confirm the box sits on the object.")
    else:
        print("matched node never projects into any frame - likely a stray cluster.")


if __name__ == "__main__":
    main()
