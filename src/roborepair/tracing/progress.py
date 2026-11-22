"""Task progress measured by the longest achieved prefix of ordered subgoals.

Task modules define the ordering; progress remains in [0, 1].
"""

from __future__ import annotations

from ..simulation.tasks import TASKS


def subgoal_status(task_id: str, state, events) -> dict:
    """Delegate to the task module's frozen subgoal logic."""
    return TASKS[task_id].subgoal_status(state, events)


def progress_score(task_id: str, status: dict) -> float:
    """Fraction of the ordered subgoal prefix achieved, in [0, 1]."""
    order = TASKS[task_id].SUBGOALS
    achieved_prefix = 0
    for name in order:
        if status.get(name, False):
            achieved_prefix += 1
        else:
            break
    return achieved_prefix / len(order)
