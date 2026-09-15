from engine.dsa.max_heap import MaxHeap


def test_empty_heap():
    heap = MaxHeap(key_fn=lambda x: x)
    assert heap.is_empty() is True
    assert len(heap) == 0
    assert heap.peek_max() is None
    assert heap.extract_max() is None


def test_insertion_and_peek():
    heap = MaxHeap(key_fn=lambda x: x)
    heap.insert(3)
    heap.insert(9)
    heap.insert(1)
    assert heap.peek_max() == 9
    assert len(heap) == 3


def test_extraction_returns_descending_order():
    heap = MaxHeap(key_fn=lambda x: x)
    for value in (5, 3, 8, 1, 9, 2):
        heap.insert(value)

    extracted = [heap.extract_max() for _ in range(6)]
    assert extracted == [9, 8, 5, 3, 2, 1]
    assert heap.is_empty() is True


def test_correct_ordering_with_key_fn():
    heap = MaxHeap(key_fn=lambda item: item["priority"])
    for priority in (2, 4, 1, 3):
        heap.insert({"priority": priority})

    order = [heap.extract_max()["priority"] for _ in range(4)]
    assert order == [4, 3, 2, 1]


def test_duplicate_priorities_are_all_retained():
    heap = MaxHeap(key_fn=lambda item: item["priority"])
    for priority in (3, 3, 3, 1):
        heap.insert({"priority": priority})

    assert heap.extract_max()["priority"] == 3
    remaining_priorities = sorted(item["priority"] for item in heap.to_list())
    assert remaining_priorities == [1, 3, 3]


def test_rebuild_after_external_mutation():
    heap = MaxHeap(key_fn=lambda item: item["value"])
    a, b, c = {"value": 1}, {"value": 2}, {"value": 3}
    for item in (a, b, c):
        heap.insert(item)

    assert heap.peek_max() is c

    a["value"] = 100
    heap.rebuild()

    assert heap.peek_max() is a
