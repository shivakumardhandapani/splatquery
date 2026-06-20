"""Turn a grounded 3D target into something a robot can act on.

navigation  -> a goal pose: where the base should stand and which way to face.
spatial Q&A -> handled in the grounding agent (this module adds geometry helpers).

The goal is a standoff pose on the open-space side of the target, at the robot's
sensor height, facing the object. We output both a 2D base goal (x, z, yaw) under
an assumed up-axis and a full 4x4 look-at pose that is up-axis agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class NavGoal:
    target_id: int
    position: np.ndarray     # (3,) world goal position for the sensor
    yaw: float               # radians, in the ground plane
    look_at: np.ndarray      # (4,4) pose that faces the target (up-axis agnostic)
    standoff: float
    score: float


def _scene_center(semantic_map) -> np.ndarray:
    if len(semantic_map) == 0:
        return np.zeros(3)
    return np.stack([n.centroid for n in semantic_map.nodes]).mean(axis=0)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right          # camera +x
    pose[:3, 1] = -true_up       # camera +y (down, OpenCV convention)
    pose[:3, 2] = fwd            # camera +z (forward)
    pose[:3, 3] = eye
    return pose


def navigation_goal(target_node, semantic_map, standoff: float = 0.8,
                    robot_height: float = 0.4, up_axis: str = "y",
                    score: float = 1.0) -> NavGoal:
    """Compute a standoff goal pose facing `target_node`."""
    up_i = _AXIS[up_axis]
    up = np.zeros(3)
    up[up_i] = 1.0

    obj = target_node.centroid.copy()
    center = _scene_center(semantic_map)

    # Horizontal approach direction (project out the up component).
    approach = obj - center
    approach[up_i] = 0.0
    if np.linalg.norm(approach) < 1e-3:           # robot is ~under/over the object
        approach = np.array([1.0, 0.0, 0.0])
        approach[up_i] = 0.0
    approach /= (np.linalg.norm(approach) + 1e-8)

    goal = obj - standoff * approach              # stand short, on the open side
    # Place at robot sensor height above the object's base.
    ground = float(target_node.bbox_min[up_i])
    goal[up_i] = ground + robot_height

    look_at = _look_at(goal, obj, up)

    # 2D yaw in the horizontal plane (the two non-up axes).
    horiz = [i for i in range(3) if i != up_i]
    d = obj - goal
    yaw = float(np.arctan2(d[horiz[1]], d[horiz[0]]))

    return NavGoal(target_node.node_id, goal, yaw, look_at, standoff, score)


def describe_goal(goal: NavGoal) -> str:
    p = goal.position
    return (f"NAV GOAL -> object {goal.target_id} "
            f"(match {goal.score:.2f})\n"
            f"  stand at: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m\n"
            f"  face yaw: {np.degrees(goal.yaw):.1f} deg, standoff {goal.standoff:.2f} m")
