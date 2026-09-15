from datetime import datetime, timezone

from engine.models.assignment import GPUAssignment


def test_assignment_can_be_created_correctly():
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assignment = GPUAssignment(assignment_id="A1", gpu_id="GPU0", user_id="U1",
                                job_id="J1", created_at=created)

    assert assignment.assignment_id == "A1"
    assert assignment.gpu_id == "GPU0"
    assert assignment.user_id == "U1"
    assert assignment.job_id == "J1"
    assert assignment.created_at == created
    assert assignment.ended_at is None
    assert assignment.is_active is True


def test_ending_an_assignment_sets_ended_at_and_flips_is_active():
    assignment = GPUAssignment(assignment_id="A1", gpu_id="GPU0", user_id="U1", job_id="J1")
    end_time = datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    assignment.end(end_time)

    assert assignment.ended_at == end_time
    assert assignment.is_active is False
