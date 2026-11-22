"""Deterministic CPU environment for pipeline checks.

The in-memory simulator implements the shared reset and execution contract.
Objects must be approached before grasping, held before lifting, and lifted
before placement. Unknown object names are reported as API errors.
"""

from __future__ import annotations

import hashlib

from .base import Event, ExecResult, InfraError, Observation, Perturbation, SimState
from .execution import compile_program
from .tasks import TASKS


class _MockSimulation:
    """Mutable sim backing the bound API. One per run_program call."""

    def __init__(self, state: SimState):
        self.state = state
        self.events: list = []
        self._gripper_over: str | None = None  # object the gripper is currently above
        self._lifted: set = set()
        self._event_clock = 0.0

    def _log(self, name, args, ok):
        self._event_clock += 1.0
        self.events.append(Event(name=name, args=tuple(args), ok=ok, ts=self._event_clock))

    # bound API
    def get_object_pose(self, name):
        ok = name in self.state.object_poses
        self._log("get_object_pose", (name,), ok)
        if not ok:
            raise KeyError(f"unknown object {name!r}")
        return self.state.object_poses[name]

    def goto_pose(self, name):
        ok = name in self.state.object_poses
        self._gripper_over = name if ok else None
        self._log("goto_pose", (name,), ok)

    def grasp(self, name):
        ok = self._gripper_over == name and self.state.grasped is None and not self.state.off_table
        if ok:
            self.state.grasped = name
        self._log("grasp", (name,), ok)

    def lift(self, name, height=0.1):
        ok = self.state.grasped == name and height > 0
        if ok:
            x, y, z = self.state.object_poses[name]
            self.state.object_poses[name] = (x, y, z + float(height))
            self._lifted.add(name)
        self._log("lift", (name, height), ok)

    def open_gripper(self):
        self.state.grasped = None
        self._log("open_gripper", (), True)

    def place_on(self, source, target):
        ok = (
            self.state.grasped == source
            and source in self._lifted
            and target in self.state.object_poses
        )
        if ok:
            tx, ty, tz = self.state.object_poses[target]
            self.state.object_poses[source] = (tx, ty, tz + 0.04)
            self.state.grasped = None
        self._log("place_on", (source, target), ok)

    def knock_off(self, name):
        ok = name in self.state.object_poses
        if ok:
            x, y, _ = self.state.object_poses[name]
            self.state.object_poses[name] = (x, y, self.state.table_z - 0.2)
            self.state.off_table = True
        self._log("knock_off", (name,), ok)


class MockRepairEnv:
    """Deterministic CPU backend for environment and pipeline checks."""

    def __init__(self, task_id: str):
        """Bind the frozen task definition for ``task_id`` (must be in TASKS)."""
        if task_id not in TASKS:
            raise ValueError(f"unknown task {task_id!r}; expected one of {sorted(TASKS)}")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.api_doc = _API_DOC
        self._sim_seed = 0
        self._init_state: SimState | None = None
        self._crash_next = False  # test hook: force an infra crash on next run_program

    def reset(self, *, sim_seed: int, perturbation: Perturbation | None = None) -> Observation:
        """Place objects deterministically from ``sim_seed`` and return the Observation."""
        import numpy as np

        self._sim_seed = sim_seed
        rng = np.random.default_rng(sim_seed)
        poses = {}
        for i, name in enumerate(self.task.OBJECT_NAMES):
            x = float(rng.uniform(-0.1, 0.1))
            y = float(rng.uniform(-0.1, 0.1)) + 0.15 * i
            poses[name] = (x, y, 0.02)
        instruction = self.task.INSTRUCTION
        object_names = list(self.task.OBJECT_NAMES)
        if perturbation is not None:
            instruction, object_names, poses = _apply_perturbation(
                perturbation, instruction, object_names, poses, rng
            )
        self._init_state = SimState(
            object_poses=dict(poses), grasped=None, table_z=0.0, off_table=False
        )
        bounds = np.array([[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.5]], dtype=np.float64)
        h = hashlib.sha256(repr(sorted(poses.items())).encode()).hexdigest()[:16]
        return Observation(
            instruction=instruction,
            object_names=object_names,
            workspace_bounds=bounds,
            initial_state_hash=h,
        )

    def trigger_infra_crash(self) -> None:
        """Test hook: the next run_program raises InfraError from inside model code."""
        self._crash_next = True

    def run_program(self, code: str, *, timeout_s: float) -> ExecResult:
        """Execute generated code in a forked process with a timeout and memory cap.

        Model failures are returned as execution results; infrastructure errors propagate.
        """
        if self._init_state is None:
            raise RuntimeError("reset() must be called before run_program()")
        import multiprocessing

        # Recreate the reset state so attempts are independent.
        state = SimState(
            object_poses=dict(self._init_state.object_poses),
            grasped=self._init_state.grasped,
            table_z=self._init_state.table_z,
            off_table=self._init_state.off_table,
        )
        crash_next = self._crash_next
        self._crash_next = False
        if crash_next:
            # run a program that touches the API so the simulated infra crash fires
            code = "get_object_pose('red_cube')\n"

        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        def _child(conn):
            try:
                import resource

                # Limit additional address space so inherited mappings do not exhaust the cap.
                with open("/proc/self/status") as fh:
                    vm_kb = next(int(line.split()[1]) for line in fh if line.startswith("VmSize"))
                cap = vm_kb * 1024 + (4 << 30)
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except Exception:  # pragma: no cover - platform dependent
                pass
            try:
                import os as _os

                # Discard generated stdout to bound output from runaway programs.
                devnull = _os.open(_os.devnull, _os.O_WRONLY)
                _os.dup2(devnull, 1)
                _os.dup2(devnull, 2)
            except Exception:  # pragma: no cover - platform dependent
                pass
            sim = _MockSimulation(state)
            api_ns = {
                "get_object_pose": sim.get_object_pose,
                "goto_pose": sim.goto_pose,
                "grasp": sim.grasp,
                "lift": sim.lift,
                "open_gripper": sim.open_gripper,
                "place_on": sim.place_on,
                "knock_off": sim.knock_off,
            }
            if crash_next:

                def raise_invalid_api_error(*a, **k):
                    raise InfraError("simulated MuJoCo process crash")

                api_ns["get_object_pose"] = raise_invalid_api_error
            payload = {"infra": None, "raised": None, "api_ok": True}
            try:
                exec(compile(code, "<program>", "exec"), {**api_ns, "__name__": "__main__"})
            except InfraError as exc:
                payload["infra"] = str(exc)
            except BaseException as exc:  # noqa: BLE001 - model errors are data
                payload["raised"] = (type(exc).__name__, str(exc))
                payload["api_ok"] = False
            payload["events"] = sim.events
            payload["state"] = sim.state
            conn.send(payload)
            conn.close()

        compile_ok, _syntax_err = compile_program(code)
        if not compile_ok:
            return ExecResult(
                compile_ok=False,
                api_ok=False,
                timed_out=False,
                raised=None,
                events=[],
                final_state=state,
            )

        proc = ctx.Process(target=_child, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        payload = None
        if parent_conn.poll(timeout_s):
            try:
                payload = parent_conn.recv()
            except EOFError:  # child died before sending
                payload = None
        if payload is None:
            proc.kill()
            proc.join(5)
            # Killed on deadline (or died silently): timed out, no events recovered.
            return ExecResult(
                compile_ok=True,
                api_ok=False,
                timed_out=True,
                raised=None,
                events=[],
                final_state=state,
            )
        proc.join(5)
        if payload["infra"] is not None:
            raise InfraError(payload["infra"])
        raised = None
        if payload["raised"] is not None:
            import builtins

            name, msg = payload["raised"]
            cls = getattr(builtins, name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                raised = cls(msg)
            else:
                raised = RuntimeError(f"{name}: {msg}")
        final = payload["state"]
        return ExecResult(
            compile_ok=True,
            api_ok=payload["api_ok"],
            timed_out=False,
            raised=raised,
            events=payload["events"],
            final_state=final,
        )

    def close(self) -> None:
        """No-op (the mock sim holds no external resources)."""


def _apply_perturbation(pert: Perturbation, instruction, object_names, poses, rng):
    from ..evaluation import perturbations

    return perturbations.apply(pert, instruction, object_names, poses, rng)


# Shared execution rules describe the API without example solutions.
_API_DOC = (
    "Code-as-policy API (privileged state, no perception):\n"
    "  get_object_pose(name) -> (x, y, z)   # look up an object's position;"
    " call this before using an object\n"
    "  goto_pose(name)            # move gripper above object\n"
    "  grasp(name)                # close gripper on object (goto_pose(name) must come first)\n"
    "  lift(name, height=0.1)     # raise grasped object\n"
    "  open_gripper()             # release the grasped object\n"
    "  place_on(source, target)   # place the lifted source onto target and release it\n"
    "\n"
    "Execution rules:\n"
    "  - These functions are ALREADY DEFINED in the execution environment. Do not import\n"
    "    anything and do not define or redefine any of these functions yourself.\n"
    "  - Top-level code runs directly; if you wrap steps in a helper function you must call it.\n"
    "  - Success is checked on the FINAL state of the scene: a lifted object must still be\n"
    "    held at the end (do not call open_gripper() after lifting unless the task says to\n"
    "    release); place_on() already releases the object it places.\n"
)
