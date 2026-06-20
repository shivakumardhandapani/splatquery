# SplatQuery — language-grounded 3D scene understanding for robots

Talk to a reconstructed 3D scene in plain language and get back a **3D location**
and a **robot navigation goal**:

> "where can I find something to drink?" → object 14 at (1.32, 0.41, 2.05) m → stand at (0.78, 0.40, 1.60) m, face 122°

Open-vocabulary (no fixed label set), runs on a single consumer GPU (RTX 4070),
and swaps between a cloud LLM and an on-device LLM with one config flag.

This is the MVP (pass 1): a **discrete semantic object map** built from posed
RGB-D frames. The architecture is deliberately set up so pass 2 (a trained,
continuous **language-embedded Gaussian field**, LangSplat / FMGS style) drops
in behind the same `SemanticMap.query()` interface.

---

## Why this project exists

It sits on top of a 3D-reconstruction stack (COLMAP → SfM → 3D Gaussian
Splatting, CUDA) and adds the VLM + LLM "brain": open-vocabulary perception,
a language→geometry grounding layer, and a robot-actionable output. That
combination — perception + deep learning + enough robotics to produce an action
goal — is exactly what current VLM/VLA robotics roles screen for.

## Pipeline

```
posed RGB-D ─▶ SAM2 masks ─▶ CLIP region embeddings ─▶ lift to 3D
            ─▶ fuse detections into object nodes (the semantic map)
            ─▶ LLM grounding agent (instruction → intent + target phrases)
            ─▶ CLIP retrieval over the map
            ─▶ navigation goal pose  /  spatial Q&A
```

| Module | File | Role |
|---|---|---|
| Data | `splatquery/data/dataset.py` | Posed RGB-D loaders (Replica, nerfstudio) |
| Perception | `splatquery/perception/` | SAM2 masks + CLIP region/text embeddings |
| Mapping | `splatquery/mapping/` | Back-project to 3D, fuse into a queryable object map |
| Agent | `splatquery/agent/` | Dual LLM backend + language→target grounding |
| Robotics | `splatquery/robotics/` | Grounded target → navigation goal pose |

---

## Setup (Ubuntu + NVIDIA, e.g. RTX 4070 laptop)

```bash
# 1. environment
conda create -n splatquery python=3.10 -y && conda activate splatquery

# 2. PyTorch matching your CUDA (check `nvidia-smi`); example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. the rest
pip install -r requirements.txt

# 4. SAM 2 (from source) + a checkpoint
pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
cd ..
```

### LLM backend

- **Claude** (default): `export ANTHROPIC_API_KEY=sk-...`
- **Local** (`agent.backend=local`): run any OpenAI-compatible server, e.g.
  ```bash
  ollama serve & ollama pull qwen2.5:7b-instruct
  ```
  The local 7B model fits alongside SAM2/CLIP on a 4070 for the query stage
  (build the map first, free the GPU, then query).

---

## Get the data (dataset-first path)

Replica ships ground-truth depth, so lifting needs no extra calibration — the
cleanest first run. Download a scene (e.g. `room_0`) in the Semantic-NeRF /
iMAP layout so the folder looks like:

```
data/replica/room_0/
  results/  frame000000.jpg  depth000000.png  ...
  traj.txt
```

(See the dataset docstring in `data/dataset.py` for the exact intrinsics/scale
assumptions; tweak `dataset.depth_scale` and `K` if your export differs.)

---

## Run

```bash
# build the map once
python scripts/01_build_map.py --config config/default.yaml \
    --out runs/room0/map.pkl \
    --set dataset.root=data/replica/room_0

# ask it things (REPL)
python scripts/02_query.py --map runs/room0/map.pkl

# or a one-shot question
python scripts/02_query.py --map runs/room0/map.pkl \
    --ask "go to the chair near the window"

# visualize map + highlighted target + nav goal
python viz/visualize_map.py --map runs/room0/map.pkl --ask "the laptop"

# switch to the local LLM for any command
... --set agent.backend=local
```

---

## Your own scene (pass 1b)

1. Phone video → frames → `colmap` / `ns-process-data` → `transforms.json`.
2. `--set dataset.type=nerfstudio dataset.root=data/myroom dataset.depth_source=mono`
   (mono depth is relative-scale — align it to your COLMAP sparse cloud for
   metric nav goals; the docstring in `mapping/lifting.py` explains the catch).

## Roadmap (pass 2 — the research-depth upgrade)

- Replace the discrete object map with a **trained language-embedded Gaussian
  field** (multi-scale CLIP features distilled into 3DGS). The `SemanticMap`
  public interface (`build`, `query`, `save/load`) stays the same.
- Add **relevancy rendering**: heatmap a query directly onto the splat view.
- Add an **A\*/RRT path** on a 2D occupancy slice from the map, not just a goal pose.
- Quantize + benchmark the local LLM for **edge latency** (the on-board story).

---

## Notes

- Designed and written to run on your machine; it has not been executed in the
  environment it was authored in (no GPU/internet there). Treat the first
  `01_build_map.py` run as the smoke test and tune `dataset.stride` / intrinsics
  as needed.
- Everything downstream of `PosedFrame` is dataset-agnostic; adding a sensor or
  dataset is one new loader.
```
