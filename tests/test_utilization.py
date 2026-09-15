from datetime import datetime, timezone

import pytest

from engine.models.utilization import UtilizationObservation


def test_observation_stores_timestamp_and_utilization():
    ts = datetime(2026, 1, 1, 13, 24, 3, tzinfo=timezone.utc)
    observation = UtilizationObservation(utilization_percent=1.5, timestamp=ts, memory_used_mb=512.0)

    assert observation.utilization_percent == 1.5
    assert observation.timestamp == ts
    assert observation.memory_used_mb == 512.0


def test_observation_defaults_timestamp_to_now_when_omitted():
    before = datetime.now(timezone.utc)
    observation = UtilizationObservation(utilization_percent=10.0)
    after = datetime.now(timezone.utc)

    assert before <= observation.timestamp <= after


def test_observation_rejects_utilization_outside_valid_range():
    with pytest.raises(ValueError):
        UtilizationObservation(utilization_percent=150.0)
    with pytest.raises(ValueError):
        UtilizationObservation(utilization_percent=-1.0)
