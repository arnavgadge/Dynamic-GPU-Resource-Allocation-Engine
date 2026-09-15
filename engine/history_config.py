"""Retention limits for the project's unbounded time-series histories
(100-scenario validation, Phase 15) - `SchedulerState.events` and
`ReclaimHistory`. Both are otherwise plain, ever-growing Python lists;
a long-running session (verified with 50 real breach-to-reclaim
cycles) would otherwise accumulate history without bound. Neither
limit ever evicts *active* scheduler state (GPUs, users, jobs,
assignments) - only these two historical logs, and only their oldest
entries once the cap is exceeded.
"""

#: Maximum number of `Event` objects `SchedulerState` keeps. Oldest
#: events are evicted first, deterministically - never active state,
#: never randomly. A frontend that needs more history than this
#: retains is out of scope for this project's in-memory design (see
#: the README's "no persistence layer" note) - this cap exists to
#: bound memory, not to serve as a long-term audit log.
MAX_EVENT_HISTORY: int = 2000

#: Maximum number of reclaim `Event`s `ReclaimHistory` (the Stack -
#: DSA Structure 6) keeps. Oldest reclaims are evicted first; `undo_
#: last`/`peek_last` are unaffected by eviction, since they only ever
#: look at the most recent end.
MAX_RECLAIM_HISTORY: int = 500
