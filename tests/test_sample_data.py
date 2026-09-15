from engine.models.enums import JobStatus
from engine.sample_data import build_sample_state


def test_sample_state_has_four_gpus_and_three_users():
    state = build_sample_state()

    assert len(state.gpus) == 4
    assert len(state.users) == 3


def test_sample_state_has_one_waiting_job_for_charlie():
    state = build_sample_state()

    waiting = state.get_waiting_jobs()
    assert len(waiting) == 1
    assert waiting[0].user_id == "U3"
    assert waiting[0].status == JobStatus.WAITING


def test_sample_state_bob_holds_two_gpus():
    state = build_sample_state()

    bob_gpus = state.get_gpus_for_user("U2")
    assert {g.gpu_id for g in bob_gpus} == {"GPU1", "GPU2"}
