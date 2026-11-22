"""Robosuite execution with reset-and-retry scenes and two robot API levels.

The wrapper maps simulator actions to shared task events and success checks.
Generated programs run in forked subprocesses with time and memory limits;
infrastructure failures propagate separately from program failures.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Event, ExecResult, InfraError, Observation, Perturbation, SimState
from .execution import compile_program

LOGGER = logging.getLogger("roborepair.robosuite")

_TASK_TO_SUITE = {
    "cube_lift": "Lift",
    "cube_stack": "Stack",
    "cube_to_zone": "Lift",  # Lift scene + static target_zone pseudo-object
    "place_in_bin": "PickPlaceCan",  # target bin quadrant exposed as static 'bin'
    "nut_on_peg": "NutAssemblyRound",  # Round nut with a yaw-invariant hole on peg2.
    "open_door": "Door",  # Unlatched articulated door.
}

_SUITE_KWARGS = {
    "open_door": {"use_latch": False},
}

_INSTRUCTIONS = {
    "cube_lift": "Lift the red cube off the table.",
    "cube_stack": "Stack the red cube on top of the green cube.",
    "cube_to_zone": "Move the red cube into the target zone marked on the table.",
    "place_in_bin": "Pick up the can and place it into the bin.",
    "nut_on_peg": "Pick up the round nut and place it over the peg.",
    "open_door": "Open the door by pulling its handle.",
}

# One fixed instruction paraphrase per task; API vocabulary is unchanged.
_PARAPHRASES = {
    "cube_lift": "Pick up the red cube from the table surface and hold it up.",
    "cube_stack": "Place the red cube onto the top of the green cube.",
    "cube_to_zone": "Carry the red cube over to the marked target zone and set it down there.",
    "place_in_bin": "Grab the can and drop it into the target bin.",
    "nut_on_peg": "Grab the round nut and slide it down over the peg.",
    "open_door": "Pull the door handle to swing the door open.",
}

# Half-extents (metres) reported by get_object_pose(..., return_bbox_extent=True) and
# used for the place_on release height (source half-height + 2.5 cm drop clearance).
_EXTENTS = {
    "red_cube": (0.02, 0.02, 0.02),
    "green_cube": (0.02, 0.02, 0.02),
    "can": (0.025, 0.025, 0.06),
    "target_zone": (0.05, 0.05, 0.005),
    "bin": (0.08, 0.10, 0.05),
    "blue_distractor": (0.02, 0.02, 0.02),
    "round_nut": (0.053, 0.053, 0.01),  # outer octagon half-extent incl. handle root
    "nut_grip": (0.011, 0.011, 0.01),  # ring-wall grip point
    "peg": (0.016, 0.016, 0.10),  # pose at the peg top
    "door_handle": (0.075, 0.015, 0.02),
    "handle_start": (0.075, 0.015, 0.02),
    "door_hinge": (0.01, 0.01, 0.30),
}

# State-only objects used by success checks, omitted from the model-visible object list.
_HIDDEN_STATICS = {"handle_start"}
_DEFAULT_EXTENT = (0.02, 0.02, 0.02)

LIFT_DELTA = 0.04  # minimum rise from resting height in metres
GRASP_NEAR_RADIUS = 0.09  # event-attribution radius in metres
GOTO_TOL = 0.015  # goto convergence tolerance (m)
MAX_SERVO_STEPS = 120  # per waypoint
OSC_OUTPUT_MAX = 0.05  # metres of translation per unit action (robosuite default)

# Additional skills exposed by the high-level API.
_HIGH_LEVEL_API_DOC = (
    "  lift(name, height=0.1)               # raise the HELD object by height metres\n"
    "  place_on(source, target)             # move the HELD source above target and release it\n"
)

_API_DOC = (
    "Code-as-policy API (privileged state, position control; all positions are (x, y, z)\n"
    "tuples in metres in the world frame):\n"
    "  get_object_pose(name, return_bbox_extent=False) -> (x, y, z)\n"
    "      # object centre position; with return_bbox_extent=True returns\n"
    "      # ((x, y, z), (ex, ey, ez)) where e* are half-extents of the object\n"
    "  get_gripper_pose() -> (x, y, z)      # current gripper fingertip position\n"
    "  goto_pose(position, quat=None, z_approach=0.10)\n"
    "      # move the gripper to `position`, approaching from z_approach metres above;\n"
    "      # orientation is fixed top-down (quat is accepted but ignored)\n"
    "  open_gripper()                       # open the fingers (release)\n"
    "  close_gripper()                      # close the fingers (grasp whatever is between them)\n"
    "  check_grasped(name) -> bool          # True if `name` is currently held\n"
    "  execute_waypoints(waypoints)         # goto_pose each position in the list, in order\n"
    "\n"
    "Execution rules:\n"
    "  - These functions are ALREADY DEFINED in the execution environment. Do not import\n"
    "    anything and do not define or redefine any of these functions yourself.\n"
    "  - Top-level code runs directly; if you wrap steps in a helper function you must call it.\n"
    "  - To grasp an object: goto_pose to its centre position, then close_gripper().\n"
    "  - Success is checked on the FINAL state of the scene: a lifted object must still be\n"
    "    held at the end (do not call open_gripper() after lifting unless the task says to\n"
    "    release); to place an object, move it above the target and open_gripper().\n"
)


class _RobosuiteSession:
    """Live robosuite session bound to the code-as-policy API (one per run_program)."""

    def __init__(
        self,
        env,
        task_id: str,
        table_z: float,
        body_ids: dict,
        statics: dict | None = None,
        pose_noise=None,
        site_ids: dict | None = None,
        aliases: dict | None = None,
        engage_radius: dict | None = None,
    ):
        # Perturb observed poses only; control and event attribution use true state.
        # Forked attempts inherit the same reset-seeded noise stream.
        self._pose_noise = pose_noise
        self.env = env
        self.task_id = task_id
        self.table_z = table_z
        self.body_ids = body_ids  # name -> mujoco body id
        # Static targets are poseable but excluded from grasp and approach attribution.
        self.statics = {k: np.asarray(v, dtype=float) for k, v in (statics or {}).items()}
        # MuJoCo sites move with their parent bodies and can receive grasp events.
        self.site_ids = site_ids or {}
        # Handle grasps are attributed to the parent object.
        self.aliases = aliases or {}
        # A closed gripper near the door handle counts as engagement: fingerpad
        # contact checks can miss a hooked bar that still moves the door.
        self.engage_radius = engage_radius or {}
        self.events: list = []
        self._clock = 0.0
        self._grip_cmd = -1.0  # persistent gripper command (-1 open, +1 close)
        self._lift_logged: set = set()
        self._action_dimension = env.action_spec[0].shape[0]
        # Measure lift relative to each object's resting height, including tall objects.
        self._initial_heights = {
            n: float(np.array(env.sim.data.body_xpos[b])[2]) for n, b in self.body_ids.items()
        }
        self._initial_heights.update(
            {n: float(np.array(env.sim.data.site_xpos[s])[2]) for n, s in self.site_ids.items()}
        )

    # utils
    def _log(self, name, args, ok):
        self._clock += 1.0
        self.events.append(Event(name=name, args=tuple(args), ok=ok, ts=self._clock))

    def _gripper(self):
        g = self.env.robots[0].gripper
        return next(iter(g.values())) if isinstance(g, dict) else g

    def end_effector_position(self) -> np.ndarray:
        r = self.env.robots[0]
        sid = getattr(r, "eef_site_id", None)
        if isinstance(sid, dict):
            sid = next(iter(sid.values()))
        if sid is None:  # fallback via the gripper's grip site
            sid = self.env.sim.model.site_name2id(self._gripper().important_sites["grip_site"])
        return np.array(self.env.sim.data.site_xpos[sid])

    def _object_position(self, name) -> np.ndarray:
        if name in self.statics:
            return self.statics[name].copy()
        if name in self.site_ids:
            return np.array(self.env.sim.data.site_xpos[self.site_ids[name]])
        return np.array(self.env.sim.data.body_xpos[self.body_ids[name]])

    def _is_known_object(self, name) -> bool:
        return name in self.body_ids or name in self.statics or name in self.site_ids

    def _step(self, pos_delta):
        a = np.zeros(self._action_dimension)
        a[:3] = np.clip(pos_delta, -1.0, 1.0)
        a[-1] = self._grip_cmd
        self.env.step(a)

    def _grasp_target_geometry(self, name):
        env = self.env
        if self.task_id in ("cube_lift", "cube_to_zone"):
            return env.cube
        if self.task_id == "place_in_bin":
            return next(o for o in env.objects if o.name == "Can")
        if self.task_id == "nut_on_peg":
            return next(o for o in env.nuts if o.name == "RoundNut")
        return env.cubeA if name == "red_cube" else env.cubeB

    def _canonical_object_name(self, name):
        return self.aliases.get(name, name)

    def _check_grasp_raw(self, name) -> bool:
        name = self._canonical_object_name(name)
        if name in self.engage_radius:
            return (
                self._grip_cmd > 0
                and float(
                    np.linalg.norm(self.end_effector_position() - self._object_position(name))
                )
                < self.engage_radius[name]
            )
        try:
            return bool(
                self.env._check_grasp(
                    gripper=self._gripper(), object_geoms=self._grasp_target_geometry(name)
                )
            )
        except Exception:  # pragma: no cover - version drift guard
            return False

    def _graspable_names(self):
        return list(self.body_ids) + [n for n in self.site_ids if n in self.engage_radius]

    def _grasped_object(self):
        for name in self._graspable_names():
            if self._check_grasp_raw(name):
                return name
        return None

    def _maybe_log_lift(self):
        held = self._grasped_object()
        if held and held not in self._lift_logged:
            z = float(self._object_position(held)[2])
            rise = z - self._initial_heights.get(held, self.table_z)
            if rise > LIFT_DELTA:
                self._lift_logged.add(held)
                self._log("lift", (held, round(rise, 3)), True)

    def _servo_to(self, target: np.ndarray) -> bool:
        # Long transits need more steps; converged moves still stop early.
        dist = float(np.linalg.norm(target - self.end_effector_position()))
        max_steps = MAX_SERVO_STEPS + int(400 * dist)
        for _ in range(max_steps):
            err = target - self.end_effector_position()
            if np.linalg.norm(err) < GOTO_TOL:
                self._maybe_log_lift()
                return True
            self._step(err / OSC_OUTPUT_MAX)
        self._maybe_log_lift()
        return bool(np.linalg.norm(target - self.end_effector_position()) < GOTO_TOL * 2)

    def _resolve_near_object(self, pos: np.ndarray, radius: float):
        best, best_d = None, radius
        for name in list(self.body_ids) + list(self.site_ids):
            d = float(np.linalg.norm(self._object_position(name) - pos))
            if d < best_d:
                best, best_d = name, d
        return best

    # bound API
    def get_object_pose(self, name, return_bbox_extent=False):
        ok = self._is_known_object(name)
        self._log("get_object_pose", (name,), ok)
        if not ok:
            raise KeyError(f"unknown object {name!r}")
        pos = self._object_position(name)
        if self._pose_noise is not None:
            sigma, rng = self._pose_noise
            pos = pos + rng.normal(0.0, sigma, size=3)
        p = tuple(float(v) for v in pos)
        if return_bbox_extent:
            return p, _EXTENTS.get(name, _DEFAULT_EXTENT)
        return p

    def get_gripper_pose(self):
        self._log("get_gripper_pose", (), True)
        return tuple(float(v) for v in self.end_effector_position())

    def goto_pose(self, position, quat=None, z_approach=0.10):
        if isinstance(position, str):
            if self._is_known_object(position):
                target = self._object_position(position).copy()
            else:
                self._log("goto_pose", (position,), False)
                raise KeyError(f"unknown object {position!r}")
        else:
            try:
                target = np.array([float(position[0]), float(position[1]), float(position[2])])
            except (TypeError, ValueError, IndexError):
                self._log("goto_pose", (repr(position),), False)
                raise TypeError("goto_pose expects a position tuple (x, y, z)") from None
        ok = True
        if z_approach and float(z_approach) > 0:
            ok = self._servo_to(target + np.array([0.0, 0.0, float(z_approach)]))
        ok = self._servo_to(target) and ok
        # Use object names in events so mock and real backends share diagnosis rules.
        near = self._resolve_near_object(target, GRASP_NEAR_RADIUS)
        self._log("goto_pose", (near if near else tuple(round(float(v), 3) for v in target),), ok)

    def open_gripper(self):
        self._grip_cmd = -1.0
        for _ in range(12):
            self._step(np.zeros(3))
        self._log("open_gripper", (), True)

    def close_gripper(self):
        self._grip_cmd = 1.0
        for _ in range(15):
            self._step(np.zeros(3))
        near = self._resolve_near_object(self.end_effector_position(), GRASP_NEAR_RADIUS)
        held = self._grasped_object()
        # Attribute a nearby close to the object, resolving handle aliases.
        if near is not None:
            self._log(
                "grasp",
                (near,),
                held is not None
                and self._canonical_object_name(held) == self._canonical_object_name(near),
            )
        else:
            self._log("close_gripper", (), held is not None)
        self._maybe_log_lift()

    def check_grasped(self, name):
        if not self._is_known_object(name):
            self._log("check_grasped", (name,), False)
            raise KeyError(f"unknown object {name!r}")
        held = False if name in self.statics else self._check_grasp_raw(name)
        self._log("check_grasped", (name,), held)
        return held

    def execute_waypoints(self, waypoints):
        for wp in list(waypoints):
            self.goto_pose(wp, z_approach=0.0)

    def lift(self, name, height=0.1):
        """Raise the held object by ``height``; fail if ``name`` is not held."""
        if not self._is_known_object(name):
            self._log("lift", (name,), False)
            raise KeyError(f"unknown object {name!r}")
        held = False if name in self.statics else self._check_grasp_raw(name)
        ok = False
        if held and height and float(height) > 0:
            target = self.end_effector_position() + np.array([0.0, 0.0, float(height)])
            self._servo_to(target)
            z = float(self._object_position(name)[2])
            rise = z - self._initial_heights.get(name, self.table_z)
            ok = self._check_grasp_raw(name) and rise > LIFT_DELTA
        self._lift_logged.add(name)  # explicit event below replaces the synthetic one
        self._log("lift", (name, float(height)), ok)

    def place_on(self, source, target):
        """Move the held ``source`` above ``target`` and release it."""
        for n in (source, target):
            if not self._is_known_object(n):
                self._log("place_on", (source, target), False)
                raise KeyError(f"unknown object {n!r}")
        held = False if source in self.statics else self._check_grasp_raw(source)
        ok = False
        if held:
            tgt = self._object_position(target)
            # Align the held object's centre, accounting for an off-centre grasp.
            comp = self.end_effector_position() - self._object_position(source)
            # A 2.5 cm drop clearance limits bouncing while clearing the target.
            drop_z = _EXTENTS.get(source, _DEFAULT_EXTENT)[2] + 0.025
            # Static targets need a higher transit to clear bin walls.
            transit = drop_z + 0.20 if target in self.statics else max(0.10, drop_z + 0.05)
            self._servo_to(tgt + comp + np.array([0.0, 0.0, transit]))
            self._servo_to(tgt + comp + np.array([0.0, 0.0, drop_z]))
            if target == "peg":
                # The nut can shift in the gripper during transit. Align its live centre
                # within 4 mm of the peg axis before releasing.
                for _ in range(300):
                    src_now = self._object_position(source)
                    err = np.array(
                        [tgt[0] - src_now[0], tgt[1] - src_now[1], (tgt[2] + drop_z) - src_now[2]]
                    )
                    if float(np.linalg.norm(err)) < 0.004:
                        break
                    self._step(err / OSC_OUTPUT_MAX * 0.3)
            self._grip_cmd = -1.0
            for _ in range(12):
                self._step(np.zeros(3))
            if target == "peg":
                # Let the released nut settle down the shaft before reading the outcome.
                for _ in range(30):
                    self._step(np.zeros(3))
            src = self._object_position(source)
            if target == "peg":
                # A threaded nut settles below the advertised peg top.
                ok = (
                    abs(src[0] - tgt[0]) < 0.03
                    and abs(src[1] - tgt[1]) < 0.03
                    and src[2] < tgt[2] - 0.02
                )
            else:
                tol = max(0.04, _EXTENTS.get(target, _DEFAULT_EXTENT)[0])
                ok = (
                    abs(src[0] - tgt[0]) < tol
                    and abs(src[1] - tgt[1]) < tol
                    and src[2] > self._object_position(target)[2] - 0.02
                )
        self._log("place_on", (source, target), ok)

    def pull(self, name, dx, dy):
        """Drag an engaged object horizontally by ``(dx, dy)`` metres."""
        if not self._is_known_object(name):
            self._log("pull", (name,), False)
            raise KeyError(f"unknown object {name!r}")
        engaged = False if name in self.statics else self._check_grasp_raw(name)
        ok = False
        if engaged:
            start = self._object_position(name)
            self._servo_to(self.end_effector_position() + np.array([float(dx), float(dy), 0.0]))
            moved = float(np.linalg.norm((self._object_position(name) - start)[:2]))
            ok = moved >= 0.4 * float(np.hypot(float(dx), float(dy)))
        self._log("pull", (name, round(float(dx), 3), round(float(dy), 3)), ok)

    def api_namespace(self, tier: str = "low_level") -> dict:
        ns = {
            "get_object_pose": self.get_object_pose,
            "get_gripper_pose": self.get_gripper_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "check_grasped": self.check_grasped,
            "execute_waypoints": self.execute_waypoints,
        }
        if tier == "high_level":
            ns["lift"] = self.lift
            ns["place_on"] = self.place_on
            if self.task_id == "open_door":  # The high-level door API includes pulling.
                ns["pull"] = self.pull
        return ns

    # final state
    def final_state(self) -> SimState:
        poses = {n: tuple(float(v) for v in self._object_position(n)) for n in self.body_ids}
        held = self._grasped_object()
        off = any(p[2] < self.table_z - 0.03 for p in poses.values())
        poses.update({n: tuple(float(v) for v in self._object_position(n)) for n in self.site_ids})
        poses.update({n: tuple(float(v) for v in p) for n, p in self.statics.items()})
        return SimState(object_poses=poses, grasped=held, table_z=self.table_z, off_table=off)


class RobosuiteRepairEnv:
    """Minimal robosuite wrapper implementing the RepairEnv contract."""

    def __init__(
        self,
        task_id: str,
        *,
        sim_version: str = "robosuite",
        asset_version: str = "",
        api_tier: str = "low_level",
        pose_noise_sigma: float = 0.0,
    ):
        """Validate task and API tier and assemble the tier-specific API doc.

        The underlying robosuite env is constructed lazily on first reset.
        """
        self.pose_noise_sigma = float(pose_noise_sigma)
        if task_id not in _TASK_TO_SUITE:
            raise ValueError(f"unknown task {task_id!r}")
        if api_tier not in ("low_level", "high_level"):
            raise ValueError(
                f"unknown api_tier {api_tier!r} (expected 'low_level' or 'high_level')"
            )
        self.task_id = task_id
        self.api_tier = api_tier
        if api_tier == "high_level":
            head, rules = _API_DOC.split("\nExecution rules:", 1)
            tier_doc = _HIGH_LEVEL_API_DOC
            extra_rules = (
                "  - place_on(source, target) positions the held object on the target and\n"
                "    releases it automatically; do not call open_gripper() after place_on.\n"
            )
            if task_id == "open_door":  # The high-level door API includes pulling.
                tier_doc = tier_doc + (
                    "  pull(name, dx, dy)                   # drag the grasped object"
                    " horizontally by (dx, dy) metres\n"
                )
                extra_rules = extra_rules + (
                    "  - To open the door: goto_pose to the door_handle position,"
                    " close_gripper()\n"
                    "    to grip it, then pull('door_handle', dx, dy) to drag the"
                    " handle sideways.\n"
                    "  - The door swings horizontally about the fixed 'door_hinge': the handle"
                    " can\n"
                    "    only move PERPENDICULAR to the line from door_hinge to door_handle, and\n"
                    "    it must travel at least 0.10 m of arc to open the door. Either\n"
                    "    perpendicular sign may be the opening direction; if one pull does not\n"
                    "    move the handle, pull the opposite way.\n"
                )
            if task_id == "nut_on_peg":  # Expose the ring-wall grip point.
                extra_rules = extra_rules + (
                    "  - The round nut is an OPEN RING: closing the gripper at its centre grasps\n"
                    "    nothing. Grasp it at the 'nut_grip' point: goto_pose to the nut_grip\n"
                    "    position, then close_gripper().\n"
                )
            self.api_doc = head + tier_doc + "\nExecution rules:" + rules + extra_rules
        else:
            self.api_doc = _API_DOC
        self.sim_version = sim_version
        self._sim_seed = 0
        self._env = None
        self._statics: dict = {}
        self._sites: dict = {}
        self._aliases: dict = {}
        self._engage: dict = {}
        self._instruction = _INSTRUCTIONS[task_id]

    # env
    def _make_suite_env(self):
        import robosuite as suite

        kwargs = dict(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=False,
            control_freq=20,
            horizon=50000,
            ignore_done=True,
            hard_reset=False,
        )
        try:  # robosuite >= 1.5
            from robosuite.controllers import load_composite_controller_config

            kwargs["controller_configs"] = load_composite_controller_config(robot="Panda")
        except ImportError:  # robosuite 1.4.x
            from robosuite import load_controller_config

            kwargs["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
        kwargs.update(_SUITE_KWARGS.get(self.task_id, {}))
        try:
            return suite.make(_TASK_TO_SUITE[self.task_id], **kwargs)
        except Exception as exc:
            raise InfraError(f"robosuite env construction failed: {exc}") from exc

    def _body_ids(self, env) -> dict:
        if self.task_id in ("cube_lift", "cube_to_zone"):
            return {"red_cube": env.cube_body_id}
        if self.task_id == "place_in_bin":
            return {"can": env.obj_body_id["Can"]}
        if self.task_id == "nut_on_peg":
            return {"round_nut": env.obj_body_id["RoundNut"]}
        if self.task_id == "open_door":
            return {}  # the graspable entity is the site-backed door_handle
        return {"red_cube": env.cubeA_body_id, "green_cube": env.cubeB_body_id}

    def _table_z(self, env) -> float:
        if self.task_id == "place_in_bin":
            return float(env.bin1_pos[2])  # bin floors; 'table' = source bin surface
        try:
            return float(env.model.mujoco_arena.table_offset[2])
        except Exception:  # pragma: no cover
            return 0.8

    @staticmethod
    def _seed_placement_sampler(sampler, rng) -> None:
        if sampler is None:  # pragma: no cover - env without placement initializer
            return
        sampler.rng = rng
        for sub in getattr(sampler, "samplers", {}).values():
            sub.rng = rng

    # perturbations (task evaluation)
    def _physical_objects(self, env):
        """(name, MujocoObject) pairs for the task's physical objects."""
        if self.task_id in ("cube_lift", "cube_to_zone"):
            return [("red_cube", env.cube)]
        if self.task_id == "cube_stack":
            return [("red_cube", env.cubeA), ("green_cube", env.cubeB)]
        if self.task_id == "place_in_bin":
            return [("can", next(o for o in env.objects if o.name == "Can"))]
        if self.task_id == "nut_on_peg":
            return [("round_nut", next(o for o in env.nuts if o.name == "RoundNut"))]
        return []

    def _apply_pose_shift(self, env, sim_seed: int) -> None:
        """Shift each task object's xy off the training placement grid (seeded)."""
        from ..seeding import derive_seed

        rng = np.random.default_rng(derive_seed(sim_seed, "pose_shift", self.task_id))
        placed: list = []
        for _name, obj in self._physical_objects(env):
            qpos = np.array(env.sim.data.get_joint_qpos(obj.joints[0]))
            for _ in range(20):
                dx, dy = rng.uniform(-0.07, 0.07, size=2)
                nx = float(np.clip(qpos[0] + dx, -0.18, 0.18))
                ny = float(np.clip(qpos[1] + dy, -0.18, 0.18))
                if all(abs(nx - px) > 0.07 or abs(ny - py) > 0.07 for px, py in placed):
                    break
            placed.append((nx, ny))
            qpos[0], qpos[1] = nx, ny
            env.sim.data.set_joint_qpos(obj.joints[0], qpos)
        env.sim.forward()

    def _distractor_pos(self, sim_seed: int, table_z: float, avoid: list) -> np.ndarray:
        from ..seeding import derive_seed

        rng = np.random.default_rng(derive_seed(sim_seed, "distractor", self.task_id))
        for _ in range(20):
            x, y = rng.uniform(-0.15, 0.15, size=2)
            if all(np.hypot(x - a[0], y - a[1]) > 0.08 for a in avoid):
                break
        return np.array([x, y, table_z + 0.02])

    def _zone_pos(self, sim_seed: int, cube_xy: np.ndarray, table_z: float) -> np.ndarray:
        """Deterministic target-zone centre: seeded polar offset from the cube."""
        from ..seeding import derive_seed

        rng = np.random.default_rng(derive_seed(sim_seed, "target_zone"))
        for _ in range(20):
            ang = rng.uniform(0.0, 2.0 * np.pi)
            dist = rng.uniform(0.12, 0.18)
            x = float(np.clip(cube_xy[0] + dist * np.cos(ang), -0.20, 0.20))
            y = float(np.clip(cube_xy[1] + dist * np.sin(ang), -0.20, 0.20))
            if np.hypot(x - cube_xy[0], y - cube_xy[1]) >= 0.10:
                break
        return np.array([x, y, table_z])

    def reset(self, *, sim_seed: int, perturbation: Perturbation | None = None) -> Observation:
        """Reset the suite env to a ``sim_seed``-determined episode and return the Observation.

        Perturbations (pose_shift / paraphrase / distractor) are applied here so every
        retry attempt replays the same perturbed instance.
        """
        if perturbation is not None and perturbation.family not in (
            "pose_shift",
            "paraphrase",
            "distractor",
        ):
            raise ValueError(f"unknown perturbation family {perturbation.family!r}")
        if (
            perturbation is not None
            and perturbation.family == "pose_shift"
            and self.task_id == "place_in_bin"
        ):
            raise NotImplementedError("pose_shift is defined for table-scene tasks only")
        self._sim_seed = sim_seed
        # Reuse the simulator with soft resets to avoid model recompilation between episodes.
        if self._env is None:
            self._env = self._make_suite_env()
        try:
            # Placement samplers own generators that np.random.seed does not reset.
            self._seed_placement_sampler(
                self._env.placement_initializer, np.random.default_rng(sim_seed)
            )
            np.random.seed(sim_seed)  # any remaining global-numpy draws
            self._env.reset()
        except Exception as exc:
            raise InfraError(f"robosuite reset failed: {exc}") from exc
        env = self._env
        if self.task_id == "nut_on_peg":
            # The sampler spawns the nut above the table; settle it before recording the scene.
            settle = np.zeros(env.action_spec[0].shape[0])
            settle[-1] = -1.0
            for _ in range(30):
                env.step(settle)
        body_ids = self._body_ids(env)
        table_z = self._table_z(env)

        instruction = _INSTRUCTIONS[self.task_id]
        if perturbation is not None and perturbation.family == "pose_shift":
            self._apply_pose_shift(env, sim_seed)
        if perturbation is not None and perturbation.family == "paraphrase":
            instruction = _PARAPHRASES[self.task_id]

        statics: dict = {}
        sites: dict = {}
        aliases: dict = {}
        engage: dict = {}
        if self.task_id == "cube_to_zone":
            cube_xy = np.array(env.sim.data.body_xpos[body_ids["red_cube"]][:2])
            statics["target_zone"] = self._zone_pos(sim_seed, cube_xy, table_z)
        if self.task_id == "place_in_bin":
            # Use the nearest bin quadrant, within the Panda arm's reach.
            statics["bin"] = np.array(env.target_bin_placements[0], dtype=float)
        if self.task_id == "nut_on_peg":
            # Place the grip point on the ring wall along the finger-separation axis,
            # away from the handle. This supports a top-down grasp at any nut yaw.
            # Expose the peg top so placement clears the shaft.
            nut = next(o for o in env.nuts if o.name == "RoundNut")
            centre = np.array(env.sim.data.body_xpos[env.obj_body_id["RoundNut"]])
            hs = env.sim.model.site_name2id(nut.important_sites["handle"])
            handle = np.array(env.sim.data.site_xpos[hs])
            grip = env.robots[0].gripper
            grip = next(iter(grip.values())) if isinstance(grip, dict) else grip
            pads = [
                np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id(g)])
                for side in ("left_fingerpad", "right_fingerpad")
                for g in grip.important_geoms[side]
            ]
            axis = pads[0][:2] - pads[-1][:2]
            axis = axis / (np.linalg.norm(axis) or 1.0)
            side = -1.0 if float(np.dot(handle[:2] - centre[:2], axis)) > 0 else 1.0
            statics["nut_grip"] = centre + side * 0.0423 * np.array([axis[0], axis[1], 0.0])
            statics["peg"] = np.array(env.sim.data.body_xpos[env.peg2_body_id]) + np.array(
                [0.0, 0.0, 0.10]
            )
        if self.task_id == "open_door":
            # Store the handle's reset pose for the final-state displacement check.
            sites["door_handle"] = env.door_handle_site_id
            engage["door_handle"] = 0.08
            statics["handle_start"] = np.array(
                env.sim.data.site_xpos[env.door_handle_site_id], dtype=float
            )
            # Expose the hinge anchor at handle height to make the swing direction observable.
            jid = env.sim.model.joint_name2id(env.door.joints[0])
            anchor = np.array(env.sim.data.xanchor[jid], dtype=float)
            statics["door_hinge"] = np.array([anchor[0], anchor[1], statics["handle_start"][2]])
        if perturbation is not None and perturbation.family == "distractor":
            avoid = [env.sim.data.body_xpos[b][:2] for b in body_ids.values()]
            avoid += [p[:2] for p in statics.values()]
            statics["blue_distractor"] = self._distractor_pos(sim_seed, table_z, avoid)
        self._statics = statics
        self._sites = sites
        self._aliases = aliases
        self._engage = engage
        self._instruction = instruction
        # Each reset defines the noise stream inherited by forked attempts.
        from ..seeding import derive_seed as _ds

        self._noise = (
            (self.pose_noise_sigma, np.random.default_rng(_ds(sim_seed, "pose_noise")))
            if self.pose_noise_sigma > 0
            else None
        )

        poses = {n: tuple(float(v) for v in env.sim.data.body_xpos[b]) for n, b in body_ids.items()}
        poses.update(
            {n: tuple(float(v) for v in env.sim.data.site_xpos[s]) for n, s in sites.items()}
        )
        poses.update({n: tuple(float(v) for v in p) for n, p in statics.items()})
        import hashlib

        h = hashlib.sha256(repr(sorted(poses.items())).encode()).hexdigest()[:16]
        bounds = np.array([[-0.4, 0.4], [-0.4, 0.4], [table_z, table_z + 0.5]], dtype=np.float64)
        return Observation(
            instruction=instruction,
            object_names=sorted(
                n for n in list(body_ids) + list(sites) + list(statics) if n not in _HIDDEN_STATICS
            ),
            workspace_bounds=bounds,
            initial_state_hash=h,
        )

    # runner
    def run_program(self, code: str, *, timeout_s: float) -> ExecResult:
        """Execute generated code in a forked subprocess (hard kill on deadline).

        The child inherits a copy-on-write copy of the live MuJoCo session, steps its
        own copy, and sends back (events, final SimState, flags); the parent's sim is
        untouched (the next episode resets anyway). Same isolation contract as the mock.
        """
        if self._env is None:
            raise RuntimeError("reset() must be called before run_program()")
        import multiprocessing

        compile_ok, _err = compile_program(code)
        env = self._env
        body_ids = self._body_ids(env)
        table_z = self._table_z(env)
        init_poses = {
            n: tuple(float(v) for v in env.sim.data.body_xpos[b]) for n, b in body_ids.items()
        }
        init_poses.update(
            {
                n: tuple(float(v) for v in env.sim.data.site_xpos[s])
                for n, s in getattr(self, "_sites", {}).items()
            }
        )
        init_poses.update({n: tuple(float(v) for v in p) for n, p in self._statics.items()})
        init_state = SimState(
            object_poses=init_poses, grasped=None, table_z=table_z, off_table=False
        )
        if not compile_ok:
            return ExecResult(
                compile_ok=False,
                api_ok=False,
                timed_out=False,
                raised=None,
                events=[],
                final_state=init_state,
            )

        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        task_id = self.task_id
        api_tier = self.api_tier
        statics = self._statics
        sites = getattr(self, "_sites", {})
        aliases = getattr(self, "_aliases", {})
        engage = getattr(self, "_engage", {})
        pose_noise = getattr(self, "_noise", None)

        def _child(conn):
            try:
                import resource

                # Limit additional address space; inherited CUDA mappings can already exceed 17 GB.
                with open("/proc/self/status") as fh:
                    vm_kb = next(int(line.split()[1]) for line in fh if line.startswith("VmSize"))
                cap = vm_kb * 1024 + (8 << 30)
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except Exception:
                pass
            try:
                import os as _os

                devnull = _os.open(_os.devnull, _os.O_WRONLY)
                _os.dup2(devnull, 1)
                _os.dup2(devnull, 2)
            except Exception:
                pass
            sim = _RobosuiteSession(
                env,
                task_id,
                table_z,
                body_ids,
                statics,
                pose_noise=pose_noise,
                site_ids=sites,
                aliases=aliases,
                engage_radius=engage,
            )
            payload = {"infra": None, "raised": None, "api_ok": True}
            try:
                exec(
                    compile(code, "<program>", "exec"),
                    {**sim.api_namespace(api_tier), "__name__": "__main__"},
                )
            except InfraError as exc:
                payload["infra"] = str(exc)
            except BaseException as exc:  # noqa: BLE001 - model errors are data
                payload["raised"] = (type(exc).__name__, str(exc))
                payload["api_ok"] = False
            try:
                sim._maybe_log_lift()
                payload["events"] = sim.events
                payload["state"] = sim.final_state()
            except Exception as exc:  # state readback must not lose the run
                payload["infra"] = payload["infra"] or f"state readback failed: {exc}"
                payload["events"] = sim.events
                payload["state"] = init_state
            conn.send(payload)
            conn.close()

        proc = ctx.Process(target=_child, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        payload = None
        got_eof = False
        if parent_conn.poll(timeout_s):
            try:
                payload = parent_conn.recv()
            except EOFError:  # pipe closed: child died before sending
                got_eof = True
        if payload is None:
            proc.kill()
            proc.join(5)
            if got_eof:
                # Premature child death is an infrastructure failure, not a model timeout.
                raise InfraError(
                    f"execution child died before reporting (exitcode={proc.exitcode})"
                )
            return ExecResult(
                compile_ok=True,
                api_ok=False,
                timed_out=True,
                raised=None,
                events=[],
                final_state=init_state,
            )
        proc.join(5)
        if payload["infra"] is not None:
            raise InfraError(payload["infra"])
        raised = None
        if payload["raised"] is not None:
            import builtins

            name, msg = payload["raised"]
            cls = getattr(builtins, name, None)
            raised = (
                cls(msg)
                if isinstance(cls, type) and issubclass(cls, BaseException)
                else RuntimeError(f"{name}: {msg}")
            )
        return ExecResult(
            compile_ok=True,
            api_ok=payload["api_ok"],
            timed_out=False,
            raised=raised,
            events=payload["events"],
            final_state=payload["state"],
        )

    def close(self) -> None:
        """Close and drop the underlying robosuite env."""
        if self._env is not None:
            self._env.close()
            self._env = None


GOLD_PROGRAMS = {
    "cube_lift": (
        "p = get_object_pose('red_cube')\n"
        "goto_pose((p[0], p[1], p[2]))\n"
        "close_gripper()\n"
        "goto_pose((p[0], p[1], p[2] + 0.15), z_approach=0)\n"
    ),
    "cube_stack": (
        "pa = get_object_pose('red_cube')\n"
        "pb = get_object_pose('green_cube')\n"
        "goto_pose((pa[0], pa[1], pa[2]))\n"
        "close_gripper()\n"
        "goto_pose((pa[0], pa[1], pa[2] + 0.15), z_approach=0)\n"
        "goto_pose((pb[0], pb[1], pb[2] + 0.05), z_approach=0.10)\n"
        "open_gripper()\n"
    ),
}

# Low-level reference programs use motion and gripper commands only.
# Transit heights clear bin walls; release heights match the high-level placement skill.
LOW_LEVEL_REFERENCE_PROGRAMS = {
    "cube_to_zone": (
        "p = get_object_pose('red_cube')\n"
        "z = get_object_pose('target_zone')\n"
        "goto_pose((p[0], p[1], p[2]))\n"
        "close_gripper()\n"
        "goto_pose((p[0], p[1], p[2] + 0.15), z_approach=0)\n"
        "goto_pose((z[0], z[1], z[2] + 0.15), z_approach=0)\n"
        "goto_pose((z[0], z[1], z[2] + 0.045), z_approach=0)\n"
        "open_gripper()\n"
    ),
    "place_in_bin": (
        "p = get_object_pose('can')\n"
        "b = get_object_pose('bin')\n"
        "goto_pose((p[0], p[1], p[2]))\n"
        "close_gripper()\n"
        "goto_pose((p[0], p[1], p[2] + 0.20), z_approach=0)\n"
        "goto_pose((b[0], b[1], b[2] + 0.285), z_approach=0)\n"
        "goto_pose((b[0], b[1], b[2] + 0.085), z_approach=0)\n"
        "open_gripper()\n"
    ),
}

LOW_LEVEL_FAULT_PROGRAMS = {
    "cube_to_zone": {
        "invalid_api": (
            "lift('red_cube', height=0.15)\n",
            "invalid_api",
        ),  # skill unavailable in the low-level API
        "no_api_calls": (
            "def go():\n    p = get_object_pose('red_cube')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "missing_grasp": (
            "p = get_object_pose('red_cube')\nz = get_object_pose('target_zone')\n"
            "goto_pose(p)\ngoto_pose((z[0], z[1], z[2] + 0.05), z_approach=0)\nopen_gripper()\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            "p = get_object_pose('red_cube')\n"
            "goto_pose((p[0] + 0.05, p[1], p[2]))\nclose_gripper()\n",
            "failed_grasp",
        ),
        "not_lifted": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nclose_gripper()\n",
            "object_not_lifted",
        ),
        "drop_in_place": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nclose_gripper()\n"
            "goto_pose((p[0], p[1], p[2] + 0.15), z_approach=0)\nopen_gripper()\n",
            "bad_placement",
        ),
    },
    "place_in_bin": {
        "invalid_api": (
            "place_on('can', 'bin')\n",
            "invalid_api",
        ),  # skill unavailable in the low-level API
        "no_api_calls": (
            "def go():\n    p = get_object_pose('can')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "missing_grasp": (
            "p = get_object_pose('can')\nb = get_object_pose('bin')\n"
            "goto_pose(p)\ngoto_pose((b[0], b[1], b[2] + 0.085), z_approach=0)\nopen_gripper()\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            "p = get_object_pose('can')\ngoto_pose((p[0] + 0.06, p[1], p[2]))\nclose_gripper()\n",
            "failed_grasp",
        ),
        "not_lifted": (
            "p = get_object_pose('can')\ngoto_pose(p)\nclose_gripper()\n",
            "object_not_lifted",
        ),
        "drop_in_place": (
            "p = get_object_pose('can')\ngoto_pose(p)\nclose_gripper()\n"
            "goto_pose((p[0], p[1], p[2] + 0.15), z_approach=0)\nopen_gripper()\n",
            "bad_placement",
        ),
    },
}

# High-level fault programs paired with their expected primary failure labels.
HIGH_LEVEL_FAULT_PROGRAMS = {
    "cube_lift": {
        "invalid_api": ("move_arm('red_cube')\n", "invalid_api"),
        "no_api_calls": (
            "def go():\n    p = get_object_pose('red_cube')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "unknown_object": ("p = get_object_pose('blue_cube')\n", "target_not_localized"),
        "missing_grasp": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nlift('red_cube', height=0.15)\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            "p = get_object_pose('red_cube')\n"
            "goto_pose((p[0] + 0.05, p[1], p[2]))\nclose_gripper()\n"
            "lift('red_cube', height=0.15)\n",
            "failed_grasp",
        ),
        "not_lifted": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nclose_gripper()\n",
            "object_not_lifted",
        ),
        "released_after_lift": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nclose_gripper()\n"
            "lift('red_cube', height=0.15)\nopen_gripper()\n",
            "released_after_lift",
        ),
    },
    "cube_stack": {
        "invalid_api": ("stack('red_cube', 'green_cube')\n", "invalid_api"),
        "no_api_calls": (
            "def go():\n    p = get_object_pose('red_cube')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "missing_grasp": (
            "pa = get_object_pose('red_cube')\ngoto_pose(pa)\nplace_on('red_cube', 'green_cube')\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            "pa = get_object_pose('red_cube')\n"
            "goto_pose((pa[0] + 0.05, pa[1], pa[2]))\nclose_gripper()\n"
            "lift('red_cube', height=0.15)\n",
            "failed_grasp",
        ),
        "not_lifted_drop": (
            # Carry AWAY from the target before dropping so the bounce cannot land on
            # the target (keeps the expected label replay-invariant).
            "pa = get_object_pose('red_cube')\ngoto_pose(pa)\nclose_gripper()\n"
            "lift('red_cube', height=0.15)\n"
            "goto_pose((pa[0] - 0.10, pa[1], pa[2] + 0.15), z_approach=0)\nopen_gripper()\n",
            "bad_placement",
        ),
    },
    "cube_to_zone": {
        "invalid_api": ("push_to_zone('red_cube')\n", "invalid_api"),
        "no_api_calls": (
            "def go():\n    p = get_object_pose('red_cube')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "unknown_object": ("p = get_object_pose('blue_cube')\n", "target_not_localized"),
        "missing_grasp": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nplace_on('red_cube', 'target_zone')\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            "p = get_object_pose('red_cube')\n"
            "goto_pose((p[0] + 0.05, p[1], p[2]))\nclose_gripper()\n"
            "lift('red_cube', height=0.15)\n",
            "failed_grasp",
        ),
        "not_lifted": (
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nclose_gripper()\n",
            "object_not_lifted",
        ),
        "drop_in_place": (
            # Zone centre is >=0.10 m from the cube start; dropping in place cannot
            # bounce into the zone (tolerance 0.05).
            "p = get_object_pose('red_cube')\ngoto_pose(p)\nclose_gripper()\n"
            "lift('red_cube', height=0.15)\nopen_gripper()\n",
            "bad_placement",
        ),
    },
    "place_in_bin": {
        "invalid_api": ("put_in_bin('can')\n", "invalid_api"),
        "no_api_calls": (
            "def go():\n    p = get_object_pose('can')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "missing_grasp": (
            "p = get_object_pose('can')\ngoto_pose(p)\nplace_on('can', 'bin')\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            "p = get_object_pose('can')\n"
            "goto_pose((p[0] + 0.06, p[1], p[2]))\nclose_gripper()\nlift('can', height=0.15)\n",
            "failed_grasp",
        ),
        "not_lifted": (
            "p = get_object_pose('can')\ngoto_pose(p)\nclose_gripper()\n",
            "object_not_lifted",
        ),
        "drop_in_place": (
            # The target bin is ~0.7 m from the source bin; dropping in place cannot
            # land inside it.
            "p = get_object_pose('can')\ngoto_pose(p)\nclose_gripper()\n"
            "lift('can', height=0.15)\nopen_gripper()\n",
            "bad_placement",
        ),
    },
    "nut_on_peg": {
        "invalid_api": ("screw_on('round_nut', 'peg')\n", "invalid_api"),
        "no_api_calls": (
            "def go():\n    p = get_object_pose('nut_grip')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "unknown_object": ("p = get_object_pose('blue_nut')\n", "target_not_localized"),
        "missing_grasp": (
            "g = get_object_pose('nut_grip')\ngoto_pose(g)\nplace_on('round_nut', 'peg')\n",
            "missing_grasp",
        ),
        "failed_grasp": (
            # Close in mid-air 5 cm above the grip point: deterministic no-contact
            # fail, still within the 9 cm attribution radius of round_nut (6.5 cm).
            "g = get_object_pose('nut_grip')\n"
            "goto_pose((g[0], g[1], g[2] + 0.05), z_approach=0)\nclose_gripper()\n"
            "lift('round_nut', height=0.15)\n",
            "failed_grasp",
        ),
        "not_lifted": (
            "g = get_object_pose('nut_grip')\ngoto_pose(g)\nclose_gripper()\n",
            "object_not_lifted",
        ),
        "drop_in_place": (
            # The peg is >=0.25 m from the nut spawn region; dropping in place cannot
            # bounce onto it.
            "g = get_object_pose('nut_grip')\ngoto_pose(g)\nclose_gripper()\n"
            "lift('round_nut', height=0.15)\nopen_gripper()\n",
            "bad_placement",
        ),
    },
    "open_door": {
        "invalid_api": ("open_the_door('door_handle')\n", "invalid_api"),
        "no_api_calls": (
            "def go():\n    p = get_object_pose('door_handle')\n    goto_pose(p)\n",
            "no_api_calls",
        ),
        "unknown_object": ("p = get_object_pose('door_knob')\n", "target_not_localized"),
        "missing_grasp": (
            # Localize only; never approach or close: no grasp attempt in the log.
            "h = get_object_pose('door_handle')\npull('door_handle', -0.11, 0.16)\n",
            "missing_grasp",
        ),
        "grasp_no_pull": (
            "h = get_object_pose('door_handle')\ngoto_pose(h)\nclose_gripper()\n",
            "door_not_opened",
        ),
        "weak_pull": (
            "h = get_object_pose('door_handle')\ngoto_pose(h)\nclose_gripper()\n"
            "pull('door_handle', -0.02, 0.03)\n",
            "door_not_opened",
        ),
    },
}

HIGH_LEVEL_REFERENCE_PROGRAMS = {
    "cube_lift": (
        "p = get_object_pose('red_cube')\n"
        "goto_pose((p[0], p[1], p[2]))\n"
        "close_gripper()\n"
        "lift('red_cube', height=0.15)\n"
    ),
    "cube_stack": (
        "pa = get_object_pose('red_cube')\n"
        "goto_pose((pa[0], pa[1], pa[2]))\n"
        "close_gripper()\n"
        "lift('red_cube', height=0.15)\n"
        "place_on('red_cube', 'green_cube')\n"
    ),
    "cube_to_zone": (
        "p = get_object_pose('red_cube')\n"
        "z = get_object_pose('target_zone')\n"
        "goto_pose((p[0], p[1], p[2]))\n"
        "close_gripper()\n"
        "lift('red_cube', height=0.15)\n"
        "place_on('red_cube', 'target_zone')\n"
    ),
    "place_in_bin": (
        "p = get_object_pose('can')\n"
        "b = get_object_pose('bin')\n"
        "goto_pose((p[0], p[1], p[2]))\n"
        "close_gripper()\n"
        "lift('can', height=0.15)\n"
        "place_on('can', 'bin')\n"
    ),
    "nut_on_peg": (
        "g = get_object_pose('nut_grip')\n"
        "goto_pose((g[0], g[1], g[2]))\n"
        "close_gripper()\n"
        "lift('round_nut', height=0.15)\n"
        "place_on('round_nut', 'peg')\n"
    ),
    "open_door": (
        "h = get_object_pose('door_handle')\n"
        "goto_pose((h[0], h[1], h[2]))\n"
        "close_gripper()\n"
        "pull('door_handle', -0.11, 0.16)\n"
    ),
}
