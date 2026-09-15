from engine.models.enums import Priority
from engine.models.user import User


def test_user_can_be_created_correctly():
    user = User(user_id="U1", name="Alice", priority=Priority.HIGH)

    assert user.user_id == "U1"
    assert user.name == "Alice"
    assert user.priority == Priority.HIGH
    assert user.assigned_gpu_ids == []
    assert user.running_job_ids == []
    assert user.gpu_count == 0


def test_user_can_hold_multiple_gpus():
    user = User(user_id="U2", name="Bob", priority=Priority.MEDIUM,
                assigned_gpu_ids=["GPU1", "GPU2"])

    assert user.gpu_count == 2
    assert user.assigned_gpu_ids == ["GPU1", "GPU2"]


def test_priority_values_are_ordered_numerically():
    assert Priority.LOW < Priority.MEDIUM < Priority.HIGH < Priority.CRITICAL
    assert Priority.LOW == 1
    assert Priority.MEDIUM == 2
    assert Priority.HIGH == 3
    assert Priority.CRITICAL == 4
