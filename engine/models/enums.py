"""Fixed, closed sets of states used across the domain model.

Enums are used instead of raw strings anywhere the set of valid
values is known and fixed. This keeps typos out of state comparisons
and gives later phases (and the frontend) a single source of truth
for what states exist.
"""

from enum import Enum, IntEnum


class GPUStatus(Enum):
    """Current lifecycle state of a single GPU.

    These describe *what is currently true about the GPU*, not a
    scheduling decision. For example ``RECLAIMING`` means the engine
    has entered the reclamation process for this GPU - the GPU
    object itself never decides to reclaim.
    """

    ACTIVE = "ACTIVE"                # Assigned and being used productively.
    IDLE = "IDLE"                    # Not assigned to anyone right now.
    IDLE_WARNING = "IDLE_WARNING"    # Assigned, but utilization is low enough to watch.
    RECLAIMING = "RECLAIMING"        # Reclaim process has started for this GPU.
    REALLOCATING = "REALLOCATING"    # GPU is being handed to a new user/job.
    #: An admin took this GPU out of the allocatable pool on purpose
    #: (Phase 14) - the `GPU` object is kept, never deleted, so its
    #: history/identity survives; `is_gpu_available` excludes it like
    #: any other non-IDLE status.
    MAINTENANCE = "MAINTENANCE"
    #: The hardware layer reported this GPU as gone/errored (Phase 6) -
    #: distinct from MAINTENANCE (an admin's deliberate choice) and
    #: from IDLE (genuinely fine and free); set only by
    #: `Scheduler.handle_gpu_failure`.
    UNAVAILABLE = "UNAVAILABLE"


class JobStatus(Enum):
    """Lifecycle state of a submitted job.

    Deliberately minimal: a job is either waiting for a GPU, running
    on one, finished, or lost its GPU before finishing. Separate
    "interrupted" vs "reclaimed" states were considered and dropped -
    at this stage both mean the same thing to the model: the job is
    no longer running and no longer holds a GPU.
    """

    WAITING = "WAITING"      # Submitted, no GPU assigned yet.
    RUNNING = "RUNNING"      # Currently executing on an assigned GPU.
    COMPLETED = "COMPLETED"  # Finished normally.
    RECLAIMED = "RECLAIMED"  # Ended early because its GPU was taken back.
    #: Withdrawn by its own user while still WAITING, before ever
    #: holding a GPU (`Scheduler.cancel_job`, Phase 7).
    CANCELLED = "CANCELLED"


class Priority(IntEnum):
    """External, numeric priority level for a user or job.

    Priority is an INPUT to the engine - set by an admin, a deadline
    system, or company policy. The engine never computes it; it only
    reads the number and uses it (e.g. in a later allocation-score
    formula such as ``0.6 * priority + 0.4 * job_size_inverse``).

    ``IntEnum`` is used (rather than a plain string label) so later
    phases can use the value directly in arithmetic without a lookup
    table. ``CRITICAL`` sits above ``HIGH`` because the project's
    design discussion calls out critical jobs that must be scored
    ahead of everything else.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """Category of an entry in the scheduler's event log.

    ``RECLAIM`` is deliberately kept as the general "a GPU was taken
    back through the confirmation flow" category (tier breach, a
    priority preemption ask, or a partial-multi-GPU resource request -
    every one of them ends the same way, via `ReclamationEngine.
    _reclaim`). The 100-scenario validation's Phase 5 finding was
    never that completion and reclaim were confused in code - a
    normal finish already logged ``STATUS``, never ``RECLAIM`` - it
    was that the *scenario spec* assumed they were the same thing.
    ``JOB_COMPLETION``/``MANUAL_RELEASE``/``AUTOMATIC_RECLAIM``/
    ``PRIORITY_PREEMPTION``/``HARDWARE_FAILURE`` below make that
    distinction explicit in the event log itself, rather than relying
    on a reader to infer it from ``reason`` text.
    """

    SYSTEM = "SYSTEM"      # Engine lifecycle (started, stopped, config changed).
    ALLOC = "ALLOC"        # A GPU was assigned to a user/job.
    MONITOR = "MONITOR"    # A utilization observation crossed a watch point.
    PROMPT = "PROMPT"      # The engine asked a user a confirmation question.
    RESPONSE = "RESPONSE"  # A user answered a prompt.
    RECLAIM = "RECLAIM"    # A GPU was taken back from its current holder (any reason - see `reason`/`metadata`).
    STATUS = "STATUS"      # A GPU/job/user status changed (e.g. back to ACTIVE after a YES).
    REQUEST = "REQUEST"    # A user/job requested GPU capacity.
    BALANCE = "BALANCE"    # New work was rerouted to rebalance load.

    #: A job finished normally, on its own - never a reclaim, never
    #: preceded by a confirmation prompt. Replaces the generic
    #: ``STATUS`` this project's `Scheduler.complete_job` used to log.
    JOB_COMPLETION = "JOB_COMPLETION"
    #: A user released a GPU themselves without being prompted first
    #: (`Scheduler.cancel_job`/an explicit release action) - distinct
    #: from a completion (the job wasn't finished) and from a reclaim
    #: (no sustained-breach/preemption confirmation preceded it).
    MANUAL_RELEASE = "MANUAL_RELEASE"
    #: A `RECLAIM` whose cause was purely a sustained utilization
    #: breach (Tier 1/Tier 2) or an unanswered grace period - never a
    #: priority preemption or a resource request. Logged *alongside*
    #: the general `RECLAIM` event (never instead of it), so every
    #: existing `RECLAIM`-based check keeps working unchanged.
    AUTOMATIC_RECLAIM = "AUTOMATIC_RECLAIM"
    #: A `RECLAIM` caused specifically by a higher-priority arrival
    #: preempting a lower-priority holder (Phase 1) - logged alongside
    #: `RECLAIM`, same reasoning as `AUTOMATIC_RECLAIM` above.
    PRIORITY_PREEMPTION = "PRIORITY_PREEMPTION"
    #: The hardware layer reported a GPU as unavailable/errored/gone
    #: missing from a poll (Phase 6) - handled through
    #: `Scheduler.handle_gpu_failure`, never a plain `RECLAIM`.
    HARDWARE_FAILURE = "HARDWARE_FAILURE"
    #: A still-WAITING job was withdrawn by its own user
    #: (`Scheduler.cancel_job`) before ever holding a GPU.
    JOB_CANCELLED = "JOB_CANCELLED"
    #: An admin bypassed the normal confirmation flow entirely
    #: (`Scheduler.force_reclaim`, Phase 8).
    ADMIN_FORCE_RECLAIM = "ADMIN_FORCE_RECLAIM"
    #: An admin changed a still-WAITING job's priority
    #: (`Scheduler.change_job_priority`, Phase 9).
    PRIORITY_CHANGED = "PRIORITY_CHANGED"


class EngineStatus(Enum):
    """Overall run state of the scheduler engine itself."""

    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
