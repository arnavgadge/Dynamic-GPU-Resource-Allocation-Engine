"""ScenarioRegistry: discover and load scenarios by id.

The seam a future CLI, test, or API would call into to list what
demonstrations exist and load one by name - so nothing (least of all
a future React frontend) needs to hardcode scenario-selection logic;
it only needs a scenario id string.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from engine.simulation.scenario import Scenario


@dataclass(frozen=True)
class ScenarioInfo:
    """A scenario's identity, without building the (possibly large)
    `Scenario` object itself - what `list_scenarios()` returns."""

    scenario_id: str
    name: str
    description: str


class ScenarioRegistry:
    """Maps scenario id -> a zero-argument builder function.

    A builder, not a ready-made `Scenario`, is registered so that
    every `load()` call - and every `list_scenarios()` call - gets a
    brand new set of `GPU`/`User`/`Job` objects, the same reasoning as
    `Scenario.fresh_copies()`: nothing here should let one caller's
    mutations leak into another's.

    Complexity: `register`/`load` are O(1) dict operations;
    `list_scenarios` is O(s) (s = registered scenarios) since it must
    build each one to read its name/description.
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[], Scenario]] = {}

    def register(self, scenario_id: str, builder: Callable[[], Scenario]) -> None:
        if scenario_id in self._builders:
            raise ValueError(f"a scenario is already registered as {scenario_id!r}")
        self._builders[scenario_id] = builder

    def list_scenarios(self) -> List[ScenarioInfo]:
        return [
            ScenarioInfo(scenario_id=scenario.scenario_id, name=scenario.name, description=scenario.description)
            for scenario in (builder() for builder in self._builders.values())
        ]

    def load(self, scenario_id: str) -> Scenario:
        try:
            builder = self._builders[scenario_id]
        except KeyError:
            raise KeyError(f"no scenario registered with id {scenario_id!r}") from None
        return builder()

    def __contains__(self, scenario_id: str) -> bool:
        return scenario_id in self._builders

    def __len__(self) -> int:
        return len(self._builders)
