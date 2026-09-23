"""The Reclamation Engine: detects genuinely idle GPUs and, through a
confirmation flow, safely returns them to the shared pool.

    UTILIZATION THRESHOLD BREACH
            v
    SUSTAINED CONDITION CONFIRMED     (engine.reclamation.monitor.is_sustained_breach)
            v
    IDLE_WARNING                      (GPU.status)
            v
    PROMPT USER                       ("are you still using this GPU?")
            v
       +---------+----------+------------------+
       v         v          v
      YES        NO      NO RESPONSE
       v         v          v
   Reset timer  Reclaim   Wait `no_response_grace_period`
   / back off  immediately        v
                              Auto reclaim

This module never reclaims merely because one reading is low - only a
breach that `is_sustained_breach` confirms has lasted the full tier
duration triggers a prompt, and even then the GPU is not taken back
until the user says "no" or fails to answer in time. There are no
leases anywhere in this flow: a GPU freed here goes straight back to
`AllocationEngine.mark_gpu_available`, the same pool every other GPU
comes from.
"""

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from engine.allocation.engine import AllocationEngine
from engine.dsa.reclaim_history import ReclaimHistory
from engine.dsa.sliding_window import UtilizationSlidingWindow
from engine.models.enums import EventType, GPUStatus, JobStatus
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.scheduler_state import SchedulerState
from engine.models.utilization import UtilizationObservation
from engine.reclamation.decision import ReclamationAction, ReclamationDecision
from engine.reclamation.monitor import is_sustained_breach
from engine.reclamation.policy import (
    DEFAULT_RECLAMATION_POLICY,
    ConfirmationResponse,
    ReclamationPolicy,
    ReclamationTierPolicy,
)


@dataclass
class _ResourceRequestContext:
    """Who is asking for a GPU that is currently held by someone else,
    and why - carried on the `_GPUWatch` for exactly one pending prompt
    (Phase 10's "partial availability" flow, `ReclamationEngine.
    request_gpu_for_reallocation`). This is never a second scheduler:
    it only tags *why* this particular confirmation prompt was raised,
    so the prompt/respond/reclaim machinery below stays the single
    real state machine for every kind of "should this GPU be given
    up?" question the project asks.
    """

    requesting_user_id: str
    requesting_user_name: str
    requesting_job_id: str
    #: True when this ask was raised because the requester's priority
    #: genuinely outranks the current holder's (Phase 1's priority
    #: preemption - `Scheduler._request_additional_gpus_if_needed`'s
    #: priority-eligible branch), as opposed to an ordinary excess-
    #: capacity reallocation ask against an *underutilized* GPU. Purely
    #: descriptive - it only changes the `category` an event/decision
    #: trace reports, never the prompt/respond mechanics themselves.
    is_priority_preemption: bool = False


@dataclass
class _GPUWatch:
    """Per-GPU bookkeeping the reclamation engine needs between calls.

    Not part of the Phase 1 `GPU` model - this is the engine's own
    working state, one step removed from what a GPU or a frontend
    needs to know about itself.
    """

    window: UtilizationSlidingWindow

    #: Set when the user answers "yes, still using it" - breach
    #: detection then ignores every observation at or before this
    #: timestamp, which is exactly what "reset the timer" means: the
    #: GPU has to breach for the *full* tier duration all over again
    #: before it is prompted a second time.
    confirmed_at: Optional[datetime] = None

    #: The job this watch last saw holding the GPU, and when it first
    #: saw it there (Day 7). A GPU's utilization history belongs to
    #: whoever held it *at the time*: idle readings taken while the GPU
    #: was unassigned, or while a previous job (already completed) held
    #: it, are no evidence about the current holder, so breach
    #: detection only looks at readings from `holder_since` onward.
    #: Without this, a job that had just been handed a long-idle GPU
    #: was asked "are you still using this GPU?" within a minute or two.
    tracked_job_id: Optional[str] = None
    holder_since: Optional[datetime] = None

    prompt_pending: bool = False
    prompt_tier: Optional[ReclamationTierPolicy] = None
    prompted_at: Optional[datetime] = None
    request_context: Optional[_ResourceRequestContext] = None

    #: Job ids that have already asked for this GPU and been told
    #: "no, I'm keeping it". Prevents `Scheduler` from immediately
    #: re-asking the same holder again on the very next tick just
    #: because the prompt is no longer pending - a declined request
    #: stays declined for that job; a *different* waiting job may
    #: still ask.
    declined_for_job_ids: set = field(default_factory=set)

    #: When a resource-request/preemption prompt on this GPU was last
    #: *resolved* (YES or NO) - the hysteresis anchor for Phase 12's
    #: anti-thrashing guard. `None` until the first such prompt
    #: resolves; never touched by an ordinary utilization-tier prompt.
    last_request_resolved_at: Optional[datetime] = None


class ReclamationEngine:
    """Watches GPU utilization and drives the confirm-then-reclaim flow.

    Owns no data `SchedulerState` doesn't also hold - like
    `AllocationEngine`, it mirrors what it needs into Phase 2
    structures for the reasons those structures exist: one
    `UtilizationSlidingWindow` per GPU (so a sustained-breach check
    never rescans a GPU's entire lifetime of readings) and one shared
    `ReclaimHistory` stack (so the most recent reclaim is always what
    a future rollback would undo first).
    """

    def __init__(
        self,
        state: SchedulerState,
        allocation_engine: Optional[AllocationEngine] = None,
        policy: ReclamationPolicy = DEFAULT_RECLAMATION_POLICY,
    ) -> None:
        self.state = state
        self.policy = policy
        self._allocation_engine = allocation_engine

        self._watches: Dict[str, _GPUWatch] = {}
        self._history = ReclaimHistory()
        self._event_seq = itertools.count(1)

    # -- watching ---------------------------------------------------

    def _watch_for(self, gpu_id: str) -> _GPUWatch:
        watch = self._watches.get(gpu_id)
        if watch is None:
            watch = _GPUWatch(window=UtilizationSlidingWindow(self.policy.monitoring_window_duration))
            self._watches[gpu_id] = watch
        return watch

    def record_utilization(self, gpu: GPU, observation: UtilizationObservation) -> Optional[ReclamationDecision]:
        """Feed one new utilization reading in for ``gpu`` and re-evaluate it.

        This both updates the GPU's own history (`GPU.record_observation`,
        Phase 1) and this engine's per-GPU sliding window, then checks
        whether anything about the reclamation flow should change -
        a fresh sustained breach raising a prompt, or a
        still-pending prompt that has gone unanswered too long.
        Returns the decision made, if any.
        """
        gpu.record_observation(observation)
        watch = self._watch_for(gpu.gpu_id)
        watch.window.add_observation(observation)
        return self._evaluate(gpu, watch, observation.timestamp)

    def _evaluate(self, gpu: GPU, watch: _GPUWatch, now: datetime) -> Optional[ReclamationDecision]:
        if watch.prompt_pending:
            # A prompt is already outstanding - don't raise a second
            # one; only check whether this one has gone unanswered
            # long enough to auto-reclaim.
            return self._check_timeout(gpu, watch, now)

        if gpu.assigned_job_id is None:
            # Nothing assigned - there is no one to prompt and
            # nothing to reclaim. Forget the previous holder, so the
            # next job to get this GPU starts with a clean baseline.
            watch.tracked_job_id = None
            watch.holder_since = None
            return None

        if watch.tracked_job_id != gpu.assigned_job_id:
            watch.tracked_job_id = gpu.assigned_job_id
            watch.holder_since = now

        tier = self._detect_tier(watch, now)
        if tier is None:
            return None

        return self._raise_prompt(gpu, watch, tier, now)

    def _detect_tier(self, watch: _GPUWatch, now: datetime) -> Optional[ReclamationTierPolicy]:
        observations = watch.window.observations(now)
        if watch.confirmed_at is not None:
            observations = [obs for obs in observations if obs.timestamp > watch.confirmed_at]
        if watch.holder_since is not None:
            observations = [obs for obs in observations if obs.timestamp >= watch.holder_since]

        # Tier 1 first: a lower threshold held for a shorter duration
        # is the stronger, faster signal (see `ReclamationTier`'s
        # docstring) - if it already fired, there is no reason to
        # also wait out Tier 2's much longer window.
        for tier_policy in (self.policy.tier1, self.policy.tier2):
            if is_sustained_breach(observations, tier_policy.utilization_threshold_percent, tier_policy.sustained_duration):
                return tier_policy
        return None

    def check_timeouts(self, now: datetime) -> List[ReclamationDecision]:
        """Auto-reclaim any GPU whose prompt has gone unanswered too long.

        Call this periodically (or before reading new utilization
        data) so a GPU is not left indefinitely in `IDLE_WARNING` just
        because no further utilization sample happened to arrive.
        """
        decisions: List[ReclamationDecision] = []
        for gpu_id, watch in list(self._watches.items()):
            if not watch.prompt_pending:
                continue
            gpu = self.state.get_gpu(gpu_id)
            if gpu is None:
                continue
            decision = self._check_timeout(gpu, watch, now)
            if decision is not None:
                decisions.append(decision)
        return decisions

    def _check_timeout(self, gpu: GPU, watch: _GPUWatch, now: datetime) -> Optional[ReclamationDecision]:
        if not watch.prompt_pending or watch.prompted_at is None:
            return None
        if now - watch.prompted_at < self.policy.no_response_grace_period:
            return None
        return self._reclaim(
            gpu, watch, watch.prompt_tier, now, reason_suffix="no response within the grace period",
            request_context=watch.request_context,
        )

    # -- confirmation flow --------------------------------------------

    def _raise_prompt(self, gpu: GPU, watch: _GPUWatch, tier: ReclamationTierPolicy, now: datetime) -> ReclamationDecision:
        watch.prompt_pending = True
        watch.prompt_tier = tier
        watch.prompted_at = now
        gpu.status = GPUStatus.IDLE_WARNING

        reason = (
            f"{tier.tier.value} ({tier.label}): utilization sustained below "
            f"{tier.utilization_threshold_percent:.0f}% for at least {tier.sustained_duration}"
        )
        event = self._log_event(
            gpu, EventType.PROMPT, now, reason=reason,
            message=f"{gpu.gpu_id}: are you still using this GPU?",
        )
        return ReclamationDecision(
            timestamp=now, gpu_id=gpu.gpu_id, tier=tier.tier,
            action=ReclamationAction.PROMPTED, reason=reason, event=event,
        )

    def prompt_for_job_completion(self, gpu: GPU, now: datetime) -> Optional[ReclamationDecision]:
        """Raise the *same* confirmation prompt the sustained-utilization
        tiers raise, but triggered by a job's estimated duration having
        elapsed rather than a utilization breach (Phase 9's "estimated
        completion" check - `Scheduler.check_estimated_completions`).

        This does not duplicate the confirmation mechanism - it is a
        second *entry point* into the exact same `watch`/prompt/
        `respond`/timeout state machine `_raise_prompt` uses. YES, NO,
        and no-response afterward are handled completely identically
        either way; `tier` is simply `None` here (already an optional
        field everywhere it's read - see `_reclaim`), so a decision
        raised this way is honestly reported as "not a utilization
        tier" rather than faking one.

        Returns ``None`` (does nothing) if a prompt is already pending
        for this GPU (e.g. a tier got there first) or the GPU isn't
        actually assigned to a job right now.
        """
        watch = self._watch_for(gpu.gpu_id)
        if watch.prompt_pending or gpu.assigned_job_id is None:
            return None

        watch.prompt_pending = True
        watch.prompt_tier = None
        watch.prompted_at = now
        gpu.status = GPUStatus.IDLE_WARNING

        reason = "estimated job duration elapsed - confirmation requested"
        event = self._log_event(
            gpu, EventType.PROMPT, now, reason=reason,
            message=f"{gpu.gpu_id}: estimated work complete - are you still using this GPU?",
        )
        return ReclamationDecision(
            timestamp=now, gpu_id=gpu.gpu_id, tier=None,
            action=ReclamationAction.PROMPTED, reason=reason, event=event,
        )

    def request_gpu_for_reallocation(
        self, gpu: GPU, requesting_user_id: str, requesting_user_name: str, requesting_job_id: str, now: datetime,
        is_priority_preemption: bool = False,
    ) -> Optional[ReclamationDecision]:
        """Raise the *same* confirmation prompt every other reclamation
        trigger raises, but on behalf of another user's waiting job
        that needs this GPU right now (Phase 10's "partial GPU
        availability" flow: a multi-GPU request can't be fully
        satisfied from genuinely free GPUs alone, so the Scheduler
        asks whoever holds one of the remaining candidates to give it
        up) - the third entry point into the one prompt/respond/
        timeout state machine every tier and `prompt_for_job_completion`
        already share, never a second, parallel notification system.

        The *current* holder always gets the real choice: YES keeps
        their GPU exactly like backing off a utilization prompt does;
        NO releases it through the same `_reclaim` this engine already
        uses for a low-utilization breach. Which job actually receives
        a GPU freed this way is decided afterward, by the ordinary
        `Scheduler.try_allocate_all` policy - this method never hands
        the GPU to ``requesting_job_id`` directly, so a higher-priority
        job that arrived in the meantime can still win it instead.

        Returns ``None`` (raises nothing) if a prompt is already
        pending for this GPU or it is not currently assigned to
        anyone - there is nothing to ask for.
        """
        watch = self._watch_for(gpu.gpu_id)
        if watch.prompt_pending or gpu.assigned_job_id is None:
            return None

        watch.prompt_pending = True
        watch.prompt_tier = None
        watch.prompted_at = now
        watch.request_context = _ResourceRequestContext(
            requesting_user_id=requesting_user_id,
            requesting_user_name=requesting_user_name,
            requesting_job_id=requesting_job_id,
            is_priority_preemption=is_priority_preemption,
        )
        gpu.status = GPUStatus.IDLE_WARNING

        if is_priority_preemption:
            reason = f"{requesting_user_name}'s higher-priority job ({requesting_job_id}) needs this GPU"
            message = (
                f"{gpu.gpu_id}: {requesting_user_name}'s higher-priority request ({requesting_job_id}) "
                f"needs a GPU - would you release this one?"
            )
        else:
            reason = f"{requesting_user_name} requested this GPU for their own waiting job ({requesting_job_id})"
            message = (
                f"{gpu.gpu_id}: {requesting_user_name} needs a GPU for {requesting_job_id} - "
                f"would you release this one?"
            )
        event = self._log_event(
            gpu, EventType.PROMPT, now, reason=reason, message=message,
            metadata={"category": EventType.PRIORITY_PREEMPTION.value if is_priority_preemption else "RESOURCE_REQUEST"},
        )
        return ReclamationDecision(
            timestamp=now, gpu_id=gpu.gpu_id, tier=None,
            action=ReclamationAction.PROMPTED, reason=reason, event=event,
            requesting_user_id=requesting_user_id, requesting_job_id=requesting_job_id,
        )

    def respond(self, gpu_id: str, response: ConfirmationResponse, now: Optional[datetime] = None) -> ReclamationDecision:
        """Resolve a pending confirmation prompt for ``gpu_id``.

        Raises ``ValueError`` if there is no prompt currently pending
        for that GPU - answering a question that was never asked is a
        caller mistake, not a silent no-op.
        """
        gpu = self.state.get_gpu(gpu_id)
        watch = self._watches.get(gpu_id)
        if gpu is None or watch is None or not watch.prompt_pending:
            raise ValueError(f"no pending confirmation prompt for GPU {gpu_id!r}")

        now = now or datetime.now(timezone.utc)
        tier = watch.prompt_tier
        request_context = watch.request_context
        self._log_event(
            gpu, EventType.RESPONSE, now, reason=f"user responded {response.value}",
            message=f"{gpu_id}: user responded {response.value}",
        )

        if response is ConfirmationResponse.YES:
            watch.prompt_pending = False
            watch.prompt_tier = None
            watch.prompted_at = None
            watch.request_context = None
            watch.confirmed_at = now
            gpu.status = GPUStatus.ACTIVE
            if request_context is not None:
                watch.declined_for_job_ids.add(request_context.requesting_job_id)
                watch.last_request_resolved_at = now
                reason = f"user declined to release the GPU for {request_context.requesting_user_name}"
            else:
                reason = "user confirmed the GPU is still in use - reclamation timer reset"
            event = self._log_event(gpu, EventType.STATUS, now, reason=reason, message=f"{gpu_id}: back to ACTIVE")
            return ReclamationDecision(
                timestamp=now, gpu_id=gpu_id, tier=tier,
                action=ReclamationAction.BACKED_OFF, reason=reason, event=event,
                requesting_user_id=request_context.requesting_user_id if request_context else None,
                requesting_job_id=request_context.requesting_job_id if request_context else None,
            )

        if request_context is not None:
            reason_suffix = f"user released the GPU for {request_context.requesting_user_name}"
        else:
            reason_suffix = "user confirmed the GPU is no longer needed"
        return self._reclaim(gpu, watch, tier, now, reason_suffix=reason_suffix, request_context=request_context)

    # -- reclaiming ------------------------------------------------------

    def _reclaim(
        self, gpu: GPU, watch: _GPUWatch, tier: Optional[ReclamationTierPolicy], now: datetime, reason_suffix: str,
        request_context: Optional[_ResourceRequestContext] = None,
    ) -> ReclamationDecision:
        job = self.state.get_job(gpu.assigned_job_id) if gpu.assigned_job_id else None
        user_id = gpu.assigned_user_id
        user = self.state.get_user(user_id) if user_id else None
        previous_status = gpu.status.value

        is_preemption = request_context is not None and request_context.is_priority_preemption
        requeue_needed = False
        if job is not None:
            if gpu.gpu_id in job.assigned_gpu_ids:
                job.assigned_gpu_ids.remove(gpu.gpu_id)
            # A multi-GPU job losing one of several GPUs is not the
            # same as it finishing (Phase 10's explicit caution: don't
            # treat the whole job as complete over one underutilized
            # GPU) - only change its top-level status once it holds
            # none at all. A single-GPU job (`gpu_count == 1`, the
            # default) behaves exactly as before: its one GPU is
            # always its last.
            if not job.assigned_gpu_ids:
                if is_preemption:
                    # Day 9: preemption is not the same claim as
                    # reclamation ("this GPU looks abandoned") - the
                    # preempted job may still genuinely need GPU
                    # capacity, it simply lost a scheduling contest.
                    # RECLAIMED previously meant it was silently
                    # dropped from the waiting system entirely; it now
                    # goes back to WAITING and re-enters the real FIFO
                    # (`requeue_job`, idempotent - see Day 5) to
                    # compete again under the ordinary policy, exactly
                    # like the brief's "requeue A if it still requires
                    # resources" - never a duplicate job, never a
                    # special second queue.
                    #
                    # Its FCFS/aging clock restarts from this moment
                    # (`submitted_at = now`, `started_at` cleared) -
                    # not left at its original arrival time. Without
                    # this, a job that had already been RUNNING for a
                    # while would re-enter the queue looking like the
                    # single longest-waiting/most-aged candidate, and
                    # (under equal-size FCFS in particular) could
                    # immediately win the very GPU it was just
                    # preempted from back from the job that preempted
                    # it - defeating the preemption entirely. Job id,
                    # user id, priority, size, and remaining GPU
                    # requirement are all otherwise untouched.
                    job.status = JobStatus.WAITING
                    job.submitted_at = now
                    job.started_at = None
                    requeue_needed = True
                else:
                    job.status = JobStatus.RECLAIMED
        if user is not None:
            if gpu.gpu_id in user.assigned_gpu_ids:
                user.assigned_gpu_ids.remove(gpu.gpu_id)
            if job is not None and job.job_id in user.running_job_ids:
                user.running_job_ids.remove(job.job_id)

        assignment = self.state.get_active_assignment_for_gpu(gpu.gpu_id)
        if assignment is not None:
            assignment.end(now)

        gpu.assigned_user_id = None
        gpu.assigned_job_id = None
        gpu.status = GPUStatus.IDLE

        watch.prompt_pending = False
        watch.prompt_tier = None
        watch.prompted_at = None
        watch.confirmed_at = now
        if request_context is not None:
            watch.last_request_resolved_at = now
        watch.request_context = None

        # `category` distinguishes *why* this RECLAIM happened
        # (Phase 5 of the 100-scenario validation's fix set) without
        # ever logging a second event or changing `event_type` itself -
        # every existing check for `EventType.RECLAIM` (count, filter,
        # etc.) keeps seeing exactly the events it always did.
        if request_context is not None:
            category = (
                EventType.PRIORITY_PREEMPTION.value if request_context.is_priority_preemption
                else "RESOURCE_REQUEST"
            )
            if is_preemption:
                # The affected user's own notification (Day 9) - backend
                # state/event, never only a frontend-invented string.
                reason = (
                    f"your GPU allocation is being reclaimed for a higher-priority scheduling "
                    f"request from {request_context.requesting_user_name} ({request_context.requesting_job_id}) "
                    f"- {reason_suffix}"
                )
            else:
                reason = f"resource request from {request_context.requesting_user_name} - {reason_suffix}"
        else:
            category = EventType.AUTOMATIC_RECLAIM.value
            tier_label = tier.tier.value if tier is not None else "UNSPECIFIED"
            reason = f"{tier_label} sustained breach - {reason_suffix}"
        user_label = user.name if user is not None else user_id
        event = self._log_event(
            gpu, EventType.RECLAIM, now, reason=reason,
            message=f"Reclaimed {gpu.gpu_id} from {user_label}",
            user_id=user_id, job_id=job.job_id if job is not None else None,
            metadata={
                "category": category,
                "previous_status": previous_status,
                "resulting_status": GPUStatus.IDLE.value,
            },
        )
        self._history.record_reclaim(event)

        if self._allocation_engine is not None:
            self._allocation_engine.mark_gpu_available(gpu.gpu_id)
            if user_id is not None:
                self._allocation_engine.release_user_gpu(user_id, gpu.gpu_id)
            if requeue_needed and job is not None:
                self._allocation_engine.requeue_job(job)
                self._log_event(
                    gpu, EventType.STATUS, now,
                    reason=f"{job.job_id} still needs {job.gpus_still_needed} more GPU(s) - returned to the waiting queue",
                    message=f"{job.job_id}: preempted, back in the waiting queue",
                    user_id=user_id, job_id=job.job_id,
                )

        return ReclamationDecision(
            timestamp=now, gpu_id=gpu.gpu_id, tier=tier.tier if tier is not None else None,
            action=ReclamationAction.RECLAIMED, reason=reason, event=event,
            requesting_user_id=request_context.requesting_user_id if request_context else None,
            requesting_job_id=request_context.requesting_job_id if request_context else None,
        )

    # -- read-only helpers -----------------------------------------------

    def has_pending_prompt(self, gpu_id: str) -> bool:
        watch = self._watches.get(gpu_id)
        return watch is not None and watch.prompt_pending

    def clear_watch_for_gpu(self, gpu_id: str, now: datetime) -> bool:
        """Unconditionally clear any pending confirmation prompt for
        ``gpu_id`` - never a reclaim, never a decision, no event
        logged. For when the GPU's entire assignment context has
        already changed hands through a path this engine had nothing
        to do with (`Scheduler.complete_job`'s normal completion,
        Phase 6's hardware-failure handling): a prompt that was
        outstanding *before* that happened is now about a job/user
        that no longer holds the GPU at all, and must never be allowed
        to later auto-reclaim (or be answered into reclaiming)
        whoever the GPU was actually reassigned to next - the exact
        bug this closes (previously reproducible: a normal completion
        left a stale grace-period timer running, which then fired
        against a brand-new, unrelated holder of the same GPU id
        several minutes later).

        Returns ``True`` if anything was actually cleared.
        """
        watch = self._watches.get(gpu_id)
        if watch is None:
            return False
        # Always restart the monitoring baseline (Day 7): the GPU's
        # assignment context just changed hands, so idle readings
        # gathered under the *previous* holder must not count towards a
        # sustained breach for whoever gets it next - even when no
        # prompt happened to be pending at that moment.
        watch.confirmed_at = now
        watch.tracked_job_id = None
        watch.holder_since = None
        if not watch.prompt_pending:
            return False
        watch.prompt_pending = False
        watch.prompt_tier = None
        watch.prompted_at = None
        watch.request_context = None
        return True

    def cancel_pending_requests_for_job(self, job_id: str, now: datetime) -> List[str]:
        """Withdraw every still-pending resource-request prompt that
        was raised on behalf of ``job_id`` (Issue 7's own real-E2E-run
        catch: a job can reach its full `gpu_count` from *one*
        candidate's release while a *second* request, raised earlier
        in the same `try_allocate_all` pass for the same deficit, is
        still awaiting a response - left alone, that second prompt
        would sit there forever asking for a GPU nobody needs anymore).

        This only clears the prompt and returns the GPU's holder to
        ACTIVE - it is not a reclaim, nothing about the GPU's
        assignment changes, and a tier-based or estimated-completion
        prompt (``request_context is None``) is never touched by this.
        Returns the GPU ids whose prompt was withdrawn, for logging.
        """
        withdrawn: List[str] = []
        for gpu_id, watch in self._watches.items():
            if not watch.prompt_pending or watch.request_context is None:
                continue
            if watch.request_context.requesting_job_id != job_id:
                continue
            gpu = self.state.get_gpu(gpu_id)
            watch.prompt_pending = False
            watch.prompt_tier = None
            watch.prompted_at = None
            watch.request_context = None
            if gpu is not None:
                gpu.status = GPUStatus.ACTIVE
                self._log_event(
                    gpu, EventType.STATUS, now,
                    reason=f"the requesting job {job_id} was fully satisfied by another GPU in the meantime",
                    message=f"{gpu_id}: resource request withdrawn - {job_id} no longer needs it",
                )
            withdrawn.append(gpu_id)
        return withdrawn

    def is_in_cooldown(self, gpu_id: str, now: datetime, cooldown: timedelta) -> bool:
        """Whether a resource-request/preemption ask on ``gpu_id`` was
        resolved too recently to ask again (Phase 12's hysteresis).
        A GPU that has never had such a prompt resolved is never in
        cooldown; an ordinary utilization-tier prompt never sets this
        at all, so it has no effect on that flow.
        """
        watch = self._watches.get(gpu_id)
        if watch is None or watch.last_request_resolved_at is None:
            return False
        return now - watch.last_request_resolved_at < cooldown

    def has_declined_for(self, gpu_id: str, job_id: str) -> bool:
        """Whether ``gpu_id``'s holder has already told ``job_id`` no."""
        watch = self._watches.get(gpu_id)
        return watch is not None and job_id in watch.declined_for_job_ids

    def pending_request_context(self, gpu_id: str) -> Optional[_ResourceRequestContext]:
        """Who is asking for ``gpu_id`` right now, if its pending prompt
        (if any) was raised by `request_gpu_for_reallocation` rather
        than a utilization tier or an estimated completion. Used only
        for read-only display (`api/serializers.py`) - never mutates
        anything.
        """
        watch = self._watches.get(gpu_id)
        return watch.request_context if watch is not None else None

    def last_reclaim(self) -> Optional[Event]:
        """The most recent reclaim `Event`, or None if nothing has been reclaimed."""
        return self._history.peek_last()

    def undo_last_reclaim(self) -> Optional[Event]:
        """Pop the most recent reclaim off the history stack.

        This only removes the record from `ReclaimHistory` - actually
        restoring the GPU/job/user state a reclaim undid is a policy
        decision left to the caller (or a later phase), same as
        Phase 2 left "what rollback means" to whoever uses the stack.
        """
        return self._history.undo_last()

    def window_for(self, gpu_id: str) -> Optional[UtilizationSlidingWindow]:
        watch = self._watches.get(gpu_id)
        return watch.window if watch is not None else None

    def _log_event(
        self, gpu: GPU, event_type: EventType, now: datetime, reason: str, message: str,
        user_id: Optional[str] = None, job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        event = Event(
            event_id=f"R{next(self._event_seq)}",
            event_type=event_type,
            timestamp=now,
            gpu_id=gpu.gpu_id,
            user_id=user_id if user_id is not None else gpu.assigned_user_id,
            job_id=job_id if job_id is not None else gpu.assigned_job_id,
            message=message,
            reason=reason,
            metadata=metadata or {},
        )
        self.state.log_event(event)
        return event
