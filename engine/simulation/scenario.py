"""Scenario: initial conditions plus a schedule of actions.

A scenario describes *state*, not decisions - the GPUs and users that
exist at the start, the jobs already waiting, and a timeline of
`ScenarioAction`s to feed into the real `Scheduler` as simulated time
passes. It is deliberately inert data: nothing on this class calls
into any engine, computes a score, or picks a GPU. `Simulator` is what
turns a `Scenario` into a running simulation.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.simulation.actions import ScenarioAction


@dataclass
class Scenario:
    """A named, deterministic simulation setup.

    ``gpus``/``users``/``initial_jobs`` are the state present the
    instant the simulation starts (before any action fires).
    ``actions`` are scheduled relative to ``start_time`` by their own
    ``offset`` and may arrive in any order in this list - the
    simulator sorts them once by offset.
    """

    scenario_id: str
    name: str
    description: str
    start_time: datetime
    gpus: List[GPU] = field(default_factory=list)
    users: List[User] = field(default_factory=list)
    initial_jobs: List[Job] = field(default_factory=list)
    actions: List[ScenarioAction] = field(default_factory=list)

    def fresh_copies(self):
        """Deep copies of every mutable piece of initial state.

        `Simulator.reset()` calls this every time - a scenario's own
        `GPU`/`User`/`Job`/action objects are never handed to a live
        `Scheduler` and mutated in place, or a second run (or a reset
        partway through the first) would start from whatever the
        *previous* run left those objects in rather than from the
        scenario's real initial conditions. This is what makes
        determinism (Section "SCENARIO DETERMINISM") and `reset()`
        actually hold.
        """
        return (
            copy.deepcopy(self.gpus),
            copy.deepcopy(self.users),
            copy.deepcopy(self.initial_jobs),
            copy.deepcopy(self.actions),
        )
