"""SplatQuery - language-grounded 3D scene understanding for robots.

Pipeline:
    posed RGB-D  ->  open-vocab 2D perception (SAM2 + CLIP)
                 ->  lift to a 3D semantic object map
                 ->  LLM grounding agent (natural language -> 3D target)
                 ->  robot navigation goal  /  spatial Q&A
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .mapping.semantic_map import SemanticMap, ObjectNode

__all__ = ["Config", "load_config", "SemanticMap", "ObjectNode"]
