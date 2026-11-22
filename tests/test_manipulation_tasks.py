"""Success boundaries and failure diagnosis on synthetic manipulation states."""

from __future__ import annotations

from roborepair.simulation.base import Event, SimState
from roborepair.simulation.tasks import cube_to_zone, place_in_bin
from roborepair.tracing.diagnosis import failure_labels


def make_event(name, args, ok=True, ts=1.0):
    return Event(name=name, args=args, ok=ok, ts=ts)


def test_cube_to_zone_success_boundary():
    """cube_to_zone succeeds only in-zone, released, and near table height."""
    zone = (0.1, 0.1, 0.8)
    ok = SimState(
        object_poses={"red_cube": (0.13, 0.08, 0.821), "target_zone": zone},
        grasped=None,
        table_z=0.8,
    )
    assert cube_to_zone.check_success(ok)
    # outside xy tolerance
    far = SimState(
        object_poses={"red_cube": (0.16, 0.1, 0.821), "target_zone": zone},
        grasped=None,
        table_z=0.8,
    )
    assert not cube_to_zone.check_success(far)
    # in zone but still held aloft
    held = SimState(
        object_poses={"red_cube": (0.1, 0.1, 0.95), "target_zone": zone},
        grasped="red_cube",
        table_z=0.8,
    )
    assert not cube_to_zone.check_success(held)


def test_place_in_bin_success_boundary():
    """place_in_bin fails when the can is outside the bin or hovering above it."""
    tgt = (0.0025, 0.1575, 0.8)
    ok = SimState(object_poses={"can": (0.03, 0.13, 0.861), "bin": tgt}, grasped=None, table_z=0.8)
    assert place_in_bin.check_success(ok)
    outside = SimState(
        object_poses={"can": (0.12, 0.13, 0.861), "bin": tgt}, grasped=None, table_z=0.8
    )
    assert not place_in_bin.check_success(outside)
    hovering = SimState(
        object_poses={"can": (0.0025, 0.1575, 0.99), "bin": tgt}, grasped=None, table_z=0.8
    )
    assert not place_in_bin.check_success(hovering)


class ExecutionOutcome:
    compile_ok = True
    api_ok = True
    timed_out = False
    raised = None

    def __init__(self, events):
        self.events = events


def test_new_task_diagnosis_cascade_uses_source_keys():
    """Dropped-in-wrong-place must diagnose bad_placement (not released_after_lift)."""
    zone = (0.1, 0.1, 0.8)
    state = SimState(
        object_poses={"red_cube": (-0.1, -0.1, 0.821), "target_zone": zone},
        grasped=None,
        table_z=0.8,
    )
    events = [
        make_event("get_object_pose", ("red_cube",)),
        make_event("goto_pose", ("red_cube",)),
        make_event("grasp", ("red_cube",)),
        make_event("lift", ("red_cube", 0.15)),
        make_event("open_gripper", ()),
    ]
    status = cube_to_zone.subgoal_status(state, events)
    assert status["source_grasped"] and status["source_lifted"] and not status["source_placed"]
    labels, _dx = failure_labels(
        "cube_to_zone", ExecutionOutcome(events), status, {}, success=False
    )
    assert labels[0] == "bad_placement"


def test_new_task_missing_grasp_label():
    """place_on without a prior grasp diagnoses missing_grasp as the top label."""
    tgt = (0.0025, 0.1575, 0.8)
    state = SimState(object_poses={"can": (0.1, -0.3, 0.86), "bin": tgt}, grasped=None, table_z=0.8)
    events = [
        make_event("get_object_pose", ("can",)),
        make_event("goto_pose", ("can",)),
        make_event("place_on", ("can", "bin"), ok=False),
    ]
    status = place_in_bin.subgoal_status(state, events)
    labels, _dx = failure_labels(
        "place_in_bin", ExecutionOutcome(events), status, {}, success=False
    )
    assert labels[0] == "missing_grasp"
