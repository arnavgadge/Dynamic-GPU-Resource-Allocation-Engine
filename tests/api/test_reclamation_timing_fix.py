"""Regression: the background auto-play clock was silently running at
~120x real speed even at "1x" (`BASE_TICK` = 1 simulated minute per
`TICK_INTERVAL_SECONDS` = 0.5 real seconds), so any 5-simulated-minute
threshold (e.g. an estimated-completion prompt) fired after ~2.5 real
seconds instead of 5 real minutes. `BACKGROUND_TICK` now matches
`TICK_INTERVAL_SECONDS` exactly, so 1x means real-time.
"""

from api.config import BACKGROUND_TICK, BASE_TICK, TICK_INTERVAL_SECONDS
from engine.simulation import load_default_registry
from api.session import SimulationSession


def test_background_tick_duration_equals_the_real_interval_between_ticks():
    """The unit/scaling fix itself: one background tick's simulated
    delta must equal the real seconds between ticks - not the
    unrelated 1-minute default `BASE_TICK` still used by manual Step.
    """
    assert BACKGROUND_TICK.total_seconds() == TICK_INTERVAL_SECONDS
    assert BASE_TICK.total_seconds() == 60  # manual Step's own default is untouched


def test_1x_background_ticks_advance_simulated_time_at_real_time_pace():
    session = SimulationSession(load_default_registry(), "interactive_demo")
    session.start()
    start = session.simulator.clock.now()

    n_ticks = 10  # represents 10 * TICK_INTERVAL_SECONDS = 5 real seconds
    for _ in range(n_ticks):
        session.background_tick()

    elapsed_simulated_seconds = (session.simulator.clock.now() - start).total_seconds()
    real_seconds_represented = n_ticks * TICK_INTERVAL_SECONDS

    assert elapsed_simulated_seconds == real_seconds_represented == 5.0


def test_five_simulated_minutes_now_takes_five_minutes_worth_of_real_ticks_not_seconds():
    """The exact bug report: a 5-simulated-minute threshold must take
    600 ticks (5 min / 0.5s per tick = 600, i.e. 300 real seconds = 5
    real minutes) to reach at 1x - not ~5-10 ticks (~2.5-5 real
    seconds), which is what the old `BASE_TICK`-based pacing gave.
    """
    session = SimulationSession(load_default_registry(), "interactive_demo")
    job = session.submit_user_request("user_a", "User A", "job", 1, 5, "HIGH")  # 5-minute job
    gpu_id = job.assigned_gpu_id
    session.start()

    reclamation = session.simulator.scheduler.reclamation_engine
    ticks_to_prompt = 0
    for _ in range(1000):
        session.background_tick()
        ticks_to_prompt += 1
        if reclamation.has_pending_prompt(gpu_id):
            break

    real_seconds_elapsed = ticks_to_prompt * TICK_INTERVAL_SECONDS
    assert reclamation.has_pending_prompt(gpu_id) is True
    # Exactly 600 ticks / 300 real seconds (5 real minutes) - matching
    # the job's own 5 simulated-minute estimated duration, not ~5-10
    # ticks (~2.5-5 real seconds) the old scaling bug produced.
    assert ticks_to_prompt == 600
    assert real_seconds_elapsed == 300.0


def test_speed_multiplier_still_scales_correctly_from_the_fixed_base():
    """5x must now mean 5x *real-time* pace, not 5x of the old
    already-inflated (~120x) pace."""
    session = SimulationSession(load_default_registry(), "interactive_demo")
    session.set_speed(5)
    session.start()
    start = session.simulator.clock.now()

    n_ticks = 10  # 10 * 0.5s = 5 real seconds elapsed
    for _ in range(n_ticks):
        session.background_tick()

    elapsed = (session.simulator.clock.now() - start).total_seconds()
    real_seconds_represented = n_ticks * TICK_INTERVAL_SECONDS
    assert elapsed == real_seconds_represented * 5  # 5x real-time pace, exactly
