from engine.dsa.linked_list import LinkedList
from engine.dsa.gpu_pool import GPUPool
from engine.models.gpu import GPU


def test_empty_list():
    ll = LinkedList()
    assert len(ll) == 0
    assert ll.is_empty() is True
    assert ll.to_list() == []


def test_single_element():
    ll = LinkedList()
    ll.append("a")
    assert len(ll) == 1
    assert ll.is_empty() is False
    assert ll.to_list() == ["a"]


def test_multiple_elements_preserve_insertion_order():
    ll = LinkedList()
    for value in ("a", "b", "c"):
        ll.append(value)
    assert ll.to_list() == ["a", "b", "c"]
    assert len(ll) == 3


def test_prepend_adds_to_front():
    ll = LinkedList()
    ll.append("b")
    ll.prepend("a")
    assert ll.to_list() == ["a", "b"]


def test_find_returns_first_match_or_none():
    ll = LinkedList()
    for value in (1, 2, 3):
        ll.append(value)
    assert ll.find(lambda v: v == 2) == 2
    assert ll.find(lambda v: v == 99) is None


def test_remove_middle_head_and_tail():
    ll = LinkedList()
    for value in ("a", "b", "c"):
        ll.append(value)

    assert ll.remove(lambda v: v == "b") is True
    assert ll.to_list() == ["a", "c"]

    assert ll.remove(lambda v: v == "a") is True
    assert ll.to_list() == ["c"]

    assert ll.remove(lambda v: v == "c") is True
    assert ll.to_list() == []
    assert ll.is_empty() is True


def test_remove_missing_value_returns_false():
    ll = LinkedList()
    ll.append("a")
    assert ll.remove(lambda v: v == "missing") is False
    assert ll.to_list() == ["a"]


def test_traversal_visits_every_element_once():
    ll = LinkedList()
    for value in range(5):
        ll.append(value)
    assert list(ll) == [0, 1, 2, 3, 4]


# -- GPUPool (domain wrapper) --------------------------------------------

def make_gpu(gpu_id: str) -> GPU:
    return GPU(gpu_id=gpu_id, total_memory_mb=24_576)


def test_gpu_pool_starts_empty():
    pool = GPUPool()
    assert pool.is_empty() is True
    assert pool.size() == 0
    assert pool.all_gpus() == []


def test_gpu_pool_add_and_traverse():
    pool = GPUPool()
    gpu0, gpu1 = make_gpu("GPU0"), make_gpu("GPU1")
    pool.add_gpu(gpu0)
    pool.add_gpu(gpu1)

    assert pool.size() == 2
    assert [g.gpu_id for g in pool] == ["GPU0", "GPU1"]


def test_gpu_pool_get_and_remove_by_id():
    pool = GPUPool()
    gpu0, gpu1 = make_gpu("GPU0"), make_gpu("GPU1")
    pool.add_gpu(gpu0)
    pool.add_gpu(gpu1)

    assert pool.get_gpu("GPU1") is gpu1
    assert pool.get_gpu("missing") is None

    assert pool.remove_gpu("GPU0") is True
    assert pool.size() == 1
    assert [g.gpu_id for g in pool.all_gpus()] == ["GPU1"]
    assert pool.remove_gpu("GPU0") is False
