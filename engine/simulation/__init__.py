"""Phase 6: the scenario simulator.

    The simulator generates inputs; the scheduler generates decisions.

`Simulator` drives a `Scenario` (initial GPUs/users/jobs plus a
timeline of `ScenarioAction`s) against a real `Scheduler` over a
`SimulationClock` that only ever advances by an exact `timedelta` -
never `time.sleep`. Nothing in this package decides which job wins,
which GPU is chosen, or when a GPU is reclaimed; every such decision
still comes from `AllocationEngine`, `ReclamationEngine`, or
`LoadBalancingRouter`, exactly as in Phases 3-5.

The module-level default registry (`DEFAULT_REGISTRY`) already has
every built-in scenario registered - `load_default_registry()` scans
this session, useful for a fresh registry in isolated tests.
"""

from engine.simulation.actions import (
    AddJobAction,
    CompleteJobAction,
    ScenarioAction,
    ScenarioActionTypes,
    UserResponseAction,
    UtilizationAction,
)
from engine.simulation.clock import SimulationClock
from engine.simulation.registry import ScenarioInfo, ScenarioRegistry
from engine.simulation.scenario import Scenario
from engine.simulation.scenarios import register_builtin_scenarios
from engine.simulation.simulator import Simulator

DEFAULT_REGISTRY = ScenarioRegistry()
register_builtin_scenarios(DEFAULT_REGISTRY)


def load_default_registry() -> ScenarioRegistry:
    """A fresh `ScenarioRegistry` with every built-in scenario
    registered - useful when a caller (e.g. a test) wants an
    isolated registry rather than sharing `DEFAULT_REGISTRY`."""
    registry = ScenarioRegistry()
    register_builtin_scenarios(registry)
    return registry


__all__ = [
    "AddJobAction",
    "CompleteJobAction",
    "ScenarioAction",
    "ScenarioActionTypes",
    "UserResponseAction",
    "UtilizationAction",
    "SimulationClock",
    "ScenarioInfo",
    "ScenarioRegistry",
    "Scenario",
    "Simulator",
    "DEFAULT_REGISTRY",
    "load_default_registry",
]
