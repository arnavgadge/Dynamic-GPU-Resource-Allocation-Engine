from engine.dsa.hashmap import HashMap
from engine.dsa.user_gpu_index import UserGPUIndex
from engine.models.assignment import GPUAssignment


def test_insertion_and_lookup():
    hm = HashMap()
    hm.put("U1", "GPU0")
    assert hm.get("U1") == "GPU0"


def test_missing_key_returns_default():
    hm = HashMap()
    assert hm.get("missing") is None
    assert hm.get("missing", "fallback") == "fallback"
    assert hm.contains("missing") is False


def test_update_overwrites_existing_key():
    hm = HashMap()
    hm.put("U1", "GPU0")
    hm.update("U1", "GPU1")
    assert hm.get("U1") == "GPU1"
    assert len(hm) == 1


def test_deletion():
    hm = HashMap()
    hm.put("U1", "GPU0")
    assert hm.remove("U1") is True
    assert hm.contains("U1") is False
    assert len(hm) == 0
    assert hm.remove("U1") is False


def test_multiple_entries_all_retrievable():
    hm = HashMap()
    for i in range(20):
        hm.put(f"U{i}", f"GPU{i}")

    assert len(hm) == 20
    for i in range(20):
        assert hm.get(f"U{i}") == f"GPU{i}"


def test_collision_handling_with_forced_small_capacity():
    # Capacity of 1 forces every key into the same bucket, exercising
    # the chaining logic directly.
    hm = HashMap(capacity=1)
    hm.put("a", 1)
    hm.put("b", 2)
    hm.put("c", 3)

    assert hm.get("a") == 1
    assert hm.get("b") == 2
    assert hm.get("c") == 3
    assert len(hm) == 3

    assert hm.remove("b") is True
    assert hm.get("b") is None
    assert hm.get("a") == 1
    assert hm.get("c") == 3


def test_resize_preserves_all_entries():
    hm = HashMap(capacity=2)
    for i in range(50):
        hm.put(i, i * i)

    assert len(hm) == 50
    for i in range(50):
        assert hm.get(i) == i * i


# -- UserGPUIndex (domain wrapper) --------------------------------------

def test_user_gpu_index_supports_multiple_gpus_per_user():
    index = UserGPUIndex()
    index.add_assignment(GPUAssignment(assignment_id="A1", gpu_id="GPU1", user_id="U2", job_id="J2"))
    index.add_assignment(GPUAssignment(assignment_id="A2", gpu_id="GPU2", user_id="U2", job_id="J3"))

    assert sorted(index.get_gpus_for_user("U2")) == ["GPU1", "GPU2"]
    assert index.get_gpus_for_user("missing") == []


def test_user_gpu_index_remove_assignment():
    index = UserGPUIndex()
    assignment = GPUAssignment(assignment_id="A1", gpu_id="GPU0", user_id="U1", job_id="J1")
    index.add_assignment(assignment)

    assert index.remove_assignment(assignment) is True
    assert index.get_gpus_for_user("U1") == []
    assert index.has_user("U1") is False
