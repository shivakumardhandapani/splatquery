"""The grounding agent.

This is the "grounding layer that converts natural language instructions into
geometric / skill-level action plans" that VLM-VLA job postings ask for.

Flow:
    1. LLM classifies the instruction (navigate vs ask) and expands the target
       concept into concrete CLIP-friendly phrases
       ("something to drink" -> ["water bottle", "mug", "soda can", ...]).
    2. Phrases are CLIP-encoded and matched against the 3D semantic map.
    3. For navigation: return the grounded target object(s).
       For questions: hand the retrieved objects + their 3D layout back to the
       LLM to compose a spatial answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .llm import LLMBackend, parse_json

_PLAN_SYSTEM = (
    "You are the language-grounding module of a robot that has a 3D semantic map "
    "of its environment. Given a user instruction, decide the intent and extract "
    "what physical object(s) the robot should locate. Be concrete and visual: the "
    "target phrases are matched against CLIP image embeddings, so prefer common, "
    "concrete noun phrases a vision model would recognize."
)

_PLAN_USER = """Instruction: "{instruction}"

Return JSON with exactly these fields:
{{
  "intent": "navigate" | "ask",
  "target": "<short canonical name of the object to find>",
  "phrases": ["<{n} concrete visual synonyms/instances of the target>"],
  "relation": "<optional spatial qualifier the user gave, e.g. 'on the table', or null>"
}}"""

_ANSWER_SYSTEM = (
    "You answer questions about a room using a list of detected objects and their "
    "3D positions (meters, world frame: x right, y up, z forward). Be concise and "
    "spatially precise. If the objects don't support an answer, say so plainly."
)


@dataclass
class GroundingResult:
    intent: str                  # "navigate" | "ask"
    target: str
    phrases: list[str]
    relation: str | None
    hits: list                   # list[QueryHit]
    answer: str | None = None    # filled for "ask" intent


class GroundingAgent:
    def __init__(self, llm: LLMBackend, encoder, semantic_map,
                 query_expansions: int = 6, top_k: int = 5):
        self.llm = llm
        self.encoder = encoder
        self.map = semantic_map
        self.n = query_expansions
        self.top_k = top_k

    def plan(self, instruction: str) -> dict:
        raw = self.llm.chat(_PLAN_SYSTEM,
                            _PLAN_USER.format(instruction=instruction, n=self.n),
                            json_mode=True)
        plan = parse_json(raw)
        phrases = plan.get("phrases") or [plan.get("target", instruction)]
        plan["phrases"] = [p for p in phrases if isinstance(p, str)][: self.n]
        return plan

    def ground(self, instruction: str) -> GroundingResult:
        plan = self.plan(instruction)
        text_emb = self.encoder.encode_text(plan["phrases"])
        hits = self.map.query(text_emb, top_k=self.top_k)
        result = GroundingResult(
            intent=plan.get("intent", "navigate"),
            target=plan.get("target", instruction),
            phrases=plan["phrases"],
            relation=plan.get("relation"),
            hits=hits,
        )
        if result.intent == "ask":
            result.answer = self._answer(instruction, hits)
        return result

    def _answer(self, instruction: str, hits) -> str:
        if not hits:
            return "I don't see anything matching that in the map."
        lines = []
        for h in hits:
            c = h.node.centroid
            lines.append(
                f"- object {h.node.node_id}: pos=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), "
                f"match={h.score:.2f}, size={np.round(h.node.extent, 2).tolist()}")
        user = (f'Question: "{instruction}"\n\nDetected objects:\n'
                + "\n".join(lines) +
                "\n\nAnswer the question using these positions.")
        return self.llm.chat(_ANSWER_SYSTEM, user)
