"""API-layer configuration - control-surface concerns only.

Nothing here is a scheduling policy value (those live in
`engine/allocation/config.py`, `engine/reclamation/policy.py`,
`engine/balancing/config.py`). This is purely "how the web adapter
drives the simulator": how fast the background clock advances per
real second, and which speed multipliers the UI is allowed to pick.
"""

from datetime import timedelta

#: Default amount of simulated time the manual "Step" control advances
#: by on a single click, when the caller doesn't specify ``minutes``
#: itself (`SimulationSession.step`). This is a discrete, per-click
#: action with no real-time correspondence to preserve - one minute
#: per click remains a convenient, unrelated default.
BASE_TICK = timedelta(minutes=1)

#: Real wall-clock seconds between background ticks while the
#: simulation is running (`SimulationSession.background_tick`).
TICK_INTERVAL_SECONDS = 0.5

#: Simulated time advanced per *background* tick at 1x, before the
#: speed multiplier is applied - kept equal to `TICK_INTERVAL_SECONDS`
#: (as a `timedelta`) so that "1x" genuinely means simulated time
#: advances at the same rate as real time: one real second, one
#: simulated second. This was previously conflated with `BASE_TICK`
#: (one simulated *minute* per 0.5 real second tick), which silently
#: ran the simulation at ~120x real speed even at "1x" - e.g. a
#: 5-simulated-minute threshold fired in ~2.5 real seconds instead of
#: 5 real minutes. Fixing the unit here changes only the *pacing* of
#: auto-running playback; it does not touch `BASE_TICK`'s own separate
#: use for the manual Step control, and it does not touch any
#: reclamation/allocation threshold value itself.
BACKGROUND_TICK = timedelta(seconds=TICK_INTERVAL_SECONDS)

#: The only speed multipliers the frontend's slider may choose. A
#: fixed, small, documented set rather than an arbitrary float, so
#: "what does the slider do" has one unambiguous answer.
ALLOWED_SPEEDS = (1, 5, 10, 50, 100)

DEFAULT_SPEED = 1

#: The company's logical GPU pool size for the interactive demo
#: (Phase 10, Requirement 2) - a *scheduler* concept ("how many GPUs
#: does the company's shared pool have"), not a claim that this many
#: physical NVIDIA cards exist on whatever machine runs this project.
#: Real hardware telemetry (`engine/hardware/`) is a completely
#: separate, optional data *source* for whichever of these logical
#: GPUs happen to be backed by a real device; see the README's
#: hardware-integration section.
COMPANY_GPU_POOL_SIZE = 10

#: The range a normal user-submitted GPU request must fall in
#: (Requirement 3): at least 1, and strictly less than the full
#: company pool - no single user request may claim the *entire*
#: pool. An admin's manual/demo assignment (`SimulationSession.
#: manual_assign_gpu`) is a separate, explicit control and is not
#: bound by this range.
MIN_USER_GPU_REQUEST = 1
MAX_USER_GPU_REQUEST = COMPANY_GPU_POOL_SIZE - 1
