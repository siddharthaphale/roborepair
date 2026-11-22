"""Stable program hashes for recorded retry attempts."""

from __future__ import annotations

from roborepair.evaluation.program_structure import (
    normalized_ast_hash,
)

CODE_A = """
pose = get_object_pose('red_cube')
goto_pose(pose)
close_gripper()
lift('red_cube', height=0.1)
"""


CODE_B = """
# grab it
p = get_object_pose('red_cube')
goto_pose(p)
close_gripper()
lift('red_cube', height=0.1000004)
"""


CODE_C = """
p = get_object_pose('red_cube')
goto_pose(p)
open_gripper()
"""


def test_normalized_ast_hash_invariance():
    assert normalized_ast_hash(CODE_A) == normalized_ast_hash(CODE_B)
    assert normalized_ast_hash(CODE_A) != normalized_ast_hash(CODE_C)
    assert normalized_ast_hash("def broken(:") is None
