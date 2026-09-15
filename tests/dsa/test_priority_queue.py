from engine.dsa.priority_queue import PriorityQueue


def test_insertion_and_peek():
    pq = PriorityQueue(key_fn=lambda x: x)
    pq.insert(3)
    pq.insert(7)
    pq.insert(1)
    assert pq.peek_best() == 7
    assert len(pq) == 3


def test_pop_returns_highest_priority_first():
    pq = PriorityQueue(key_fn=lambda x: x)
    for value in (5, 1, 9, 3):
        pq.insert(value)

    order = [pq.pop_best() for _ in range(4)]
    assert order == [9, 5, 3, 1]
    assert pq.is_empty() is True


def test_ordering_uses_custom_comparator():
    # Priority is the negative of the value -> smallest value first.
    pq = PriorityQueue(key_fn=lambda x: -x)
    for value in (5, 1, 9, 3):
        pq.insert(value)

    order = [pq.pop_best() for _ in range(4)]
    assert order == [1, 3, 5, 9]


def test_equal_priority_elements_are_returned_fifo():
    pq = PriorityQueue(key_fn=lambda item: item["priority"])
    pq.insert({"priority": 1, "name": "first"})
    pq.insert({"priority": 1, "name": "second"})
    pq.insert({"priority": 1, "name": "third"})

    order = [pq.pop_best()["name"] for _ in range(3)]
    assert order == ["first", "second", "third"]


def test_higher_priority_beats_earlier_arrival():
    pq = PriorityQueue(key_fn=lambda item: item["priority"])
    pq.insert({"priority": 1, "name": "low_but_first"})
    pq.insert({"priority": 5, "name": "high_but_second"})

    assert pq.pop_best()["name"] == "high_but_second"
    assert pq.pop_best()["name"] == "low_but_first"


def test_empty_queue_peek_and_pop_return_none():
    pq = PriorityQueue(key_fn=lambda x: x)
    assert pq.peek_best() is None
    assert pq.pop_best() is None
