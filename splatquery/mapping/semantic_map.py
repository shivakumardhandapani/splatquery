"""The 3D semantic map - the heart of SplatQuery.

Detections from all frames are fused into `ObjectNode`s: spatial clusters
(one physical object = one node) each carrying an averaged CLIP embedding.
A natural-language query becomes a CLIP text embedding; cosine similarity
against every node's embedding ranks the objects. This is the open-vocabulary
"language field" in its MVP form - a discrete object graph rather than a
per-Gaussian field. Pass 2 swaps this class's internals for a trained
continuous field without changing its public query() interface.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ObjectNode:
    node_id: int
    centroid: np.ndarray            # (3,)
    bbox_min: np.ndarray            # (3,)
    bbox_max: np.ndarray            # (3,)
    embedding: np.ndarray           # (D,) L2-normalized
    n_detections: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class QueryHit:
    node: ObjectNode
    score: float


class SemanticMap:
    def __init__(self, nodes: list[ObjectNode], clip_dim: int):
        self.nodes = nodes
        self.clip_dim = clip_dim
        self._emb = (np.stack([n.embedding for n in nodes])
                     if nodes else np.zeros((0, clip_dim), dtype=np.float32))

    # --- construction -------------------------------------------------------
    @classmethod
    def build(cls, detections, voxel_size=0.02, cluster_eps=0.10,
              cluster_min_points=30, merge_sim_threshold=0.85,
              merge_overlap_threshold=0.2) -> "SemanticMap":
        """Fuse per-frame detections into object nodes via spatial clustering."""
        from sklearn.cluster import DBSCAN  # scikit-learn is a transitive dep of open3d-ml; see README

        if not detections:
            return cls([], clip_dim=0)
        clip_dim = detections[0].embedding.shape[0]
        centroids = np.stack([d.centroid for d in detections])
        labels = DBSCAN(eps=cluster_eps, min_samples=3).fit_predict(centroids)

        nodes: list[ObjectNode] = []
        nid = 0
        for lab in sorted(set(labels)):
            if lab == -1:                      # DBSCAN noise
                continue
            members = [d for d, l in zip(detections, labels) if l == lab]
            pts = np.concatenate([m.points for m in members], axis=0)
            if pts.shape[0] < cluster_min_points:
                continue
            pts = _voxel_downsample(pts, voxel_size)
            # Confidence-weighted mean embedding (more points seen -> more weight).
            weights = np.array([m.points.shape[0] for m in members], dtype=np.float32)
            emb = np.average(np.stack([m.embedding for m in members]),
                             axis=0, weights=weights)
            emb /= (np.linalg.norm(emb) + 1e-8)
            nodes.append(ObjectNode(
                node_id=nid,
                centroid=pts.mean(axis=0),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
                embedding=emb.astype(np.float32),
                n_detections=len(members),
                points=pts.astype(np.float32),
            ))
            nid += 1

        nodes = _merge_nodes(nodes, merge_sim_threshold, cluster_eps,
                             merge_overlap_threshold)
        return cls(nodes, clip_dim)

    # --- query --------------------------------------------------------------
    def query(self, text_embeddings: np.ndarray, top_k: int = 5) -> list[QueryHit]:
        """Rank object nodes against one or more text embeddings (max over phrases)."""
        if len(self.nodes) == 0:
            return []
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings[None, :]
        sims = self._emb @ text_embeddings.T          # (n_nodes, n_phrases)
        best = sims.max(axis=1)                        # best phrase per node
        order = np.argsort(-best)[:top_k]
        return [QueryHit(self.nodes[i], float(best[i])) for i in order]

    # --- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"nodes": self.nodes, "clip_dim": self.clip_dim}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticMap":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return cls(blob["nodes"], blob["clip_dim"])

    def __len__(self) -> int:
        return len(self.nodes)


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0 or points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def _box_volume(node) -> float:
    return float(np.prod(np.clip(node.bbox_max - node.bbox_min, 1e-6, None)))


def _is_structural(node, struct_size: float) -> bool:
    """Floor / wall / ceiling / rug: spans more than `struct_size` m on >=2 axes.
    These are background surfaces, not graspable objects, and their room-sized
    boxes would otherwise swallow every object during merging - so we never
    merge them into anything."""
    ext = node.bbox_max - node.bbox_min
    return int(np.sum(ext > struct_size)) >= 2


def _containment(a, b) -> float:
    """Fraction of the smaller box's volume lying inside the intersection."""
    lo = np.maximum(a.bbox_min, b.bbox_min)
    hi = np.minimum(a.bbox_max, b.bbox_max)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    if inter <= 0.0:
        return 0.0
    return inter / min(_box_volume(a), _box_volume(b))


def _merge_nodes(nodes, sim_threshold, dist_eps, overlap_threshold,
                 struct_size=1.5, size_ratio_min=0.1):
    """Conservatively merge fragments of the same object via union-find.

    A pair is merged only if ALL hold:
      - neither node is a large structural surface (floor/wall/ceiling/rug), so
        background can never swallow objects;
      - their boxes are comparable in size (smaller >= size_ratio_min of larger),
        blocking a big node from absorbing tiny distinct objects;
      - the smaller box is substantially contained in the other (> overlap_threshold);
      - AND the two look alike (CLIP cosine > sim_threshold).
    Requiring both geometry and appearance keeps the merge safe; union-find makes
    it transitive so chained fragments collapse into one object.
    """
    n = len(nodes)
    if n == 0:
        return nodes
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        if _is_structural(nodes[i], struct_size):
            continue
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            if _is_structural(b, struct_size):
                continue
            va, vb = _box_volume(a), _box_volume(b)
            comparable = min(va, vb) / max(va, vb) > size_ratio_min
            contained = _containment(a, b) > overlap_threshold
            similar = float(a.embedding @ b.embedding) > sim_threshold
            if comparable and contained and similar:
                union(i, j)

    groups: dict[int, list] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(nodes[idx])

    merged = []
    for k, group in enumerate(groups.values()):
        if len(group) == 1:
            node = group[0]
            node.node_id = k
            merged.append(node)
            continue
        pts = np.concatenate([g.points for g in group], axis=0)
        w = np.array([g.n_detections for g in group], dtype=np.float32)
        emb = np.average(np.stack([g.embedding for g in group]), axis=0, weights=w)
        emb /= (np.linalg.norm(emb) + 1e-8)
        merged.append(ObjectNode(
            node_id=k, centroid=pts.mean(axis=0),
            bbox_min=pts.min(axis=0), bbox_max=pts.max(axis=0),
            embedding=emb.astype(np.float32),
            n_detections=int(w.sum()), points=pts.astype(np.float32)))
    return merged
