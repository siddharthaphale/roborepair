"""Task definitions: subgoals, success checks, gold + scripted fault programs."""

from . import cube_lift, cube_stack, cube_to_zone, nut_on_peg, open_door, place_in_bin

TASKS = {
    "cube_lift": cube_lift,
    "cube_stack": cube_stack,
    "cube_to_zone": cube_to_zone,
    "place_in_bin": place_in_bin,
    "nut_on_peg": nut_on_peg,
    "open_door": open_door,
}

__all__ = [
    "cube_lift",
    "cube_stack",
    "cube_to_zone",
    "place_in_bin",
    "nut_on_peg",
    "open_door",
    "TASKS",
]
