#!/usr/bin/env python
"""Visualize the semantic map; optionally highlight a query's top match
and draw the computed navigation goal.

    python viz/visualize_map.py --map runs/room0/map.pkl
    python viz/visualize_map.py --map runs/room0/map.pkl --ask "the laptop"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatquery.config import load_config
from splatquery.mapping.semantic_map import SemanticMap


def _box(node, color):
    import open3d as o3d
    aabb = o3d.geometry.AxisAlignedBoundingBox(node.bbox_min, node.bbox_max)
    aabb.color = color
    return aabb


def main():
    import open3d as o3d

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--map", required=True)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    smap = SemanticMap.load(args.map)
    geoms = []

    # All object point clouds in gray, with neutral bounding boxes.
    for node in smap.nodes:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(node.points)
        pc.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(pc)
        geoms.append(_box(node, (0.5, 0.5, 0.5)))

    if args.ask:
        from splatquery.perception.encoder import CLIPEncoder
        from splatquery.robotics.navigation import navigation_goal
        enc = CLIPEncoder(cfg.perception.clip_model, cfg.perception.clip_pretrained,
                          cfg.device, cfg.perception.crop_padding)
        hits = smap.query(enc.encode_text([args.ask]), top_k=1)
        if hits:
            node = hits[0].node
            geoms.append(_box(node, (1.0, 0.2, 0.2)))   # highlight target red
            goal = navigation_goal(node, smap, cfg.robotics.standoff,
                                   cfg.robotics.robot_height,
                                   cfg.get_dotted("robotics.up_axis", "y"))
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            sphere.translate(goal.position)
            sphere.paint_uniform_color([0.1, 0.5, 1.0])  # nav goal blue
            geoms.append(sphere)
            print(f"highlighting object {node.node_id} (score {hits[0].score:.3f})")

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
