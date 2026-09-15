from datetime import datetime, timezone

from engine.models.enums import EventType
from engine.models.event import Event


def test_event_can_represent_gpu_user_job_relationships():
    ts = datetime(2026, 1, 1, 13, 24, 3, tzinfo=timezone.utc)
    event = Event(event_id="E1", event_type=EventType.ALLOC, gpu_id="GPU3",
                   user_id="U1", job_id="J7", timestamp=ts,
                   message="Assigned GPU 3 to Alice", reason="Highest allocation score")

    assert event.event_id == "E1"
    assert event.event_type == EventType.ALLOC
    assert event.gpu_id == "GPU3"
    assert event.user_id == "U1"
    assert event.job_id == "J7"
    assert event.timestamp == ts
    assert event.message == "Assigned GPU 3 to Alice"
    assert event.reason == "Highest allocation score"


def test_event_relationships_are_all_optional():
    event = Event(event_id="E2", event_type=EventType.SYSTEM, message="Engine started")

    assert event.gpu_id is None
    assert event.user_id is None
    assert event.job_id is None
    assert event.reason is None
    assert event.metadata == {}


def test_every_event_type_can_be_represented():
    for event_type in EventType:
        event = Event(event_id="E", event_type=event_type, message="msg")
        assert event.event_type is event_type
