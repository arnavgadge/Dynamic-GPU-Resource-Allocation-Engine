import pytest

from engine.simulation import load_default_registry
from engine.simulation.registry import ScenarioRegistry
from engine.simulation.scenario import Scenario

REQUIRED_SCENARIO_IDS = {"gta5_excel", "ml_video", "multiple_ml", "idle_user", "imbalance"}


# ------------------------------------------------------------------
# Test 1 - registry returns all required scenarios
# ------------------------------------------------------------------

def test_registry_lists_every_required_scenario():
    registry = load_default_registry()
    ids = {info.scenario_id for info in registry.list_scenarios()}

    assert REQUIRED_SCENARIO_IDS.issubset(ids)


def test_registry_also_includes_the_end_to_end_scenario():
    registry = load_default_registry()
    ids = {info.scenario_id for info in registry.list_scenarios()}
    assert "full_lifecycle" in ids


def test_registry_includes_the_interactive_demo_scenario_with_a_blank_pool():
    registry = load_default_registry()
    scenario = registry.load("interactive_demo")

    assert len(scenario.gpus) == 10
    assert {u.user_id for u in scenario.users} == {"user_a", "user_b", "user_c", "user_d"}
    assert scenario.initial_jobs == []
    assert scenario.actions == []


# ------------------------------------------------------------------
# Test 2 - a scenario loads correctly
# ------------------------------------------------------------------

def test_scenario_loads_with_expected_shape():
    registry = load_default_registry()
    scenario = registry.load("gta5_excel")

    assert isinstance(scenario, Scenario)
    assert scenario.scenario_id == "gta5_excel"
    assert scenario.name
    assert scenario.description
    assert len(scenario.gpus) >= 1
    assert len(scenario.users) >= 1


def test_loading_an_unknown_scenario_raises():
    registry = load_default_registry()
    with pytest.raises(KeyError):
        registry.load("does-not-exist")


def test_registering_a_duplicate_id_raises():
    registry = ScenarioRegistry()
    registry.register("x", lambda: None)
    with pytest.raises(ValueError):
        registry.register("x", lambda: None)


def test_each_load_call_returns_independent_objects():
    registry = load_default_registry()
    first = registry.load("gta5_excel")
    second = registry.load("gta5_excel")

    assert first is not second
    assert first.gpus[0] is not second.gpus[0]
