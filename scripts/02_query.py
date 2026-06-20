#!/usr/bin/env python
"""Query a built semantic map with natural language.

Examples:
    python scripts/02_query.py --map runs/room0/map.pkl --ask "where can I find something to drink?"
    python scripts/02_query.py --map runs/room0/map.pkl            # interactive REPL

Navigation instructions ("go to the laptop") return a goal pose.
Questions ("what's on the table?") return a spatial answer.
The LLM decides which, per instruction.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatquery.agent.grounding import GroundingAgent
from splatquery.agent.llm import build_backend
from splatquery.config import load_config
from splatquery.mapping.semantic_map import SemanticMap
from splatquery.perception.encoder import CLIPEncoder
from splatquery.robotics.navigation import navigation_goal, describe_goal


def handle(agent, cfg, instruction):
    result = agent.ground(instruction)
    print(f'\nintent={result.intent}  target="{result.target}"  '
          f'phrases={result.phrases}')
    if not result.hits:
        print("  (no matching objects in the map)")
        return
    if result.intent == "ask":
        print("\nANSWER:\n" + (result.answer or ""))
    else:
        top = result.hits[0]
        goal = navigation_goal(
            top.node, agent.map,
            standoff=cfg.robotics.standoff,
            robot_height=cfg.robotics.robot_height,
            up_axis=cfg.get_dotted("robotics.up_axis", "y"),
            score=top.score)
        print("\n" + describe_goal(goal))
    print("\n  candidates:")
    for h in result.hits:
        c = h.node.centroid
        print(f"    obj {h.node.node_id:>2}  score {h.score:.3f}  "
              f"pos ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--map", required=True)
    ap.add_argument("--ask", default=None, help="single instruction; omit for REPL")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    smap = SemanticMap.load(args.map)
    encoder = CLIPEncoder(cfg.perception.clip_model, cfg.perception.clip_pretrained,
                          cfg.device, cfg.perception.crop_padding)
    agent = GroundingAgent(build_backend(cfg), encoder, smap,
                           cfg.agent.query_expansions, cfg.agent.top_k)
    print(f"[ready] map has {len(smap)} objects | LLM backend = {cfg.agent.backend}")

    if args.ask:
        handle(agent, cfg, args.ask)
        return
    while True:
        try:
            instruction = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if instruction.lower() in {"quit", "exit", "q"}:
            break
        if instruction:
            handle(agent, cfg, instruction)


if __name__ == "__main__":
    main()
