from engine.dsa.min_heap import MinHeap
from engine.dsa.gpu_utilization_heap import GPUUtilizationHeap
from engine.models.gpu import GPU
from engine.models.utilization import UtilizationObservation


def test_empty_heap():
    heap = MinHeap(key_fn=lambda x: x)
    assert heap.is_empty() is True
    assert len(heap) == 0
    assert heap.peek_min() is None
    assert heap.extract_min() is None


def test_single_element():
    heap = MinHeap(key_fn=lambda x: x)
    heap.insert(5)
    assert heap.peek_min() == 5
    assert len(heap) == 1
    assert heap.extract_min() == 5
    assert heap.is_empty() is True


def test_extraction_returns_ascending_order():
    heap = MinHeap(key_fn=lambda x: x)
    for value in (5, 3, 8, 1, 9, 2):
        heap.insert(value)

    extracted = [heap.extract_min() for _ in range(6)]
    assert extracted == [1, 2, 3, 5, 8, 9]
    assert heap.is_empty() is True


def test_peek_does_not_remove():
    heap = MinHeap(key_fn=lambda x: x)
    heap.insert(3)
    heap.insert(1)
    assert heap.peek_min() == 1
    assert heap.peek_min() == 1
    assert len(heap) == 2


def test_duplicate_keys_are_all_retained():
    heap = MinHeap(key_fn=lambda x: x)
    for value in (4, 4, 4, 1):
        heap.insert(value)

    assert heap.extract_min() == 1
    remaining = sorted(heap.to_list())
    assert remaining == [4, 4, 4]


def test_rebuild_after_external_mutation():
    heap = MinHeap(key_fn=lambda item: item["value"])
    a, b, c = {"value": 10}, {"value": 20}, {"value": 5}
    for item in (a, b, c):
        heap.insert(item)

    assert heap.peek_min() is c

    b["value"] = 1  # mutate an already-inserted item's key directly
    heap.rebuild()

    assert heap.peek_min() is b


# -- GPUUtilizationHeap (domain wrapper) --------------------------------

def make_gpu(gpu_id: str, utilization: float) -> GPU:
    gpu = GPU(gpu_id=gpu_id, total_memory_mb=24_576)
    gpu.record_observation(UtilizationObservation(utilization_percent=utilization))
    return gpu


def test_gpu_utilization_heap_finds_least_utilized():
    heap = GPUUtilizationHeap()
    heap.insert_gpu(make_gpu("GPU0", 90.0))
    heap.insert_gpu(make_gpu("GPU1", 5.0))
    heap.insert_gpu(make_gpu("GPU2", 70.0))
    heap.insert_gpu(make_gpu("GPU3", 15.0))

    least = heap.peek_least_utilized()
    assert least.gpu_id == "GPU1"
    assert heap.size() == 4


def test_gpu_utilization_heap_extract_in_ascending_order():
    heap = GPUUtilizationHeap()
    for gpu_id, util in [("GPU0", 90.0), ("GPU1", 5.0), ("GPU2", 70.0), ("GPU3", 15.0)]:
        heap.insert_gpu(make_gpu(gpu_id, util))

    order = [heap.extract_least_utilized().gpu_id for _ in range(4)]
    assert order == ["GPU1", "GPU3", "GPU2", "GPU0"]
    assert heap.is_empty() is True


def test_gpu_utilization_heap_refresh_after_utilization_changes():
    heap = GPUUtilizationHeap()
    gpu_low = make_gpu("GPU0", 5.0)
    gpu_high = make_gpu("GPU1", 90.0)
    heap.insert_gpu(gpu_low)
    heap.insert_gpu(gpu_high)

    assert heap.peek_least_utilized().gpu_id == "GPU0"

    gpu_high.record_observation(UtilizationObservation(utilization_percent=1.0))
    heap.refresh()

    assert heap.peek_least_utilized().gpu_id == "GPU1"
