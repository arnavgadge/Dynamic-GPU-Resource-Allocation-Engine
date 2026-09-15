from engine.dsa.stack import Stack
from engine.dsa.reclaim_history import ReclaimHistory
from engine.models.enums import EventType
from engine.models.event import Event


def test_empty_stack():
    s = Stack()
    assert s.is_empty() is True
    assert len(s) == 0
    assert s.peek() is None
    assert s.pop() is None


def test_push_and_peek():
    s = Stack()
    s.push("a")
    s.push("b")
    assert s.peek() == "b"
    assert len(s) == 2


def test_lifo_ordering():
    s = Stack()
    for value in ("first", "second", "third"):
        s.push(value)

    assert s.pop() == "third"
    assert s.pop() == "second"
    assert s.pop() == "first"
    assert s.is_empty() is True


def test_pop_on_empty_stack_returns_none():
    s = Stack()
    assert s.pop() is None


# -- ReclaimHistory (domain wrapper) ------------------------------------

def make_reclaim_event(event_id: str, gpu_id: str) -> Event:
    return Event(event_id=event_id, event_type=EventType.RECLAIM, gpu_id=gpu_id,
                 message=f"Reclaimed {gpu_id}")


def test_reclaim_history_pops_most_recent_first():
    history = ReclaimHistory()
    e1 = make_reclaim_event("E1", "GPU0")
    e2 = make_reclaim_event("E2", "GPU1")
    e3 = make_reclaim_event("E3", "GPU2")

    history.record_reclaim(e1)
    history.record_reclaim(e2)
    history.record_reclaim(e3)

    assert history.size() == 3
    assert history.peek_last() is e3
    assert history.undo_last() is e3
    assert history.undo_last() is e2
    assert history.undo_last() is e1
    assert history.is_empty() is True


def test_reclaim_history_undo_on_empty_returns_none():
    history = ReclaimHistory()
    assert history.undo_last() is None
