"""Scheduler: the thin orchestrator wiring Phase 3/4/5's engines together.

Every phase's own architecture diagram draws the same shape - a
"Scheduler" sitting above Allocation, Reclamation and Load Balancing,
calling into each in a fixed sequence. Until now that sequence only
existed informally (duplicated between `main.py`'s demo functions and
a test helper in `tests/balancing/test_integration.py`). This class is
that sequence, written once, so Phase 6's simulator - and any future
caller (an API, a CLI, another test) - has one real object to drive
instead of gluing three engines together itself each time.

It adds no scheduling policy of its own. Every decision (which job,
which GPU, when to reclaim) still comes from the engine that already
owns that decision - `Scheduler` only calls them, in order, and keeps
the small amount of "who talks to whom" glue in one place.
"""

import itertools
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from engine.allocation.decision import AllocationDecision, AllocationPolicy, CandidateInfo
from engine.allocation.engine import AllocationEngine
from engine.balancing.config import BalancingPolicy, DEFAULT_BALANCING_POLICY
from engine.balancing.decision import (
    LoadBalancingCandidate,
    LoadBalancingTrace,
    ReallocationPath,
    RoutingDecision,
    RoutingOutcome,
)
from engine.balancing.router import LoadBalancingRouter
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.reclamation.decision import ReclamationDecision
from engine.reclamation.engine import ReclamationEngine
from engine.reclamation.policy import ConfirmationResponse, DEFAULT_RECLAMATION_POLICY, ReclamationPolicy


class Scheduler:
    """Owns one `SchedulerState` and the three engines that act on it.

    Complexity
    ----------
    add_gpu / add_user / submit_job    same as the underlying engine call - O(1) amortized
    try_allocate_all                     O(m . (log n + k)) - m allocations actually made,
                                            each paying Phase 3's select cost (O(k)) and
                                            Phase 5's routing cost (O(n + r log r)); see
                                            those phases' README sections for the O(k)/O(n)
                                            terms themselves.
    record_utilization                     O(1) amortized + O(w) - see Phase 4's README
    respond_to_prompt / check_reclamation_timeouts   O(1) / O(g) - see Phase 4's README
    complete_job                             O(1) - dict lookups and small list removals
    """

    def __init__(
        self,
        state: Optional[SchedulerState] = None,
        reclamation_policy: ReclamationPolicy = DEFAULT_RECLAMATION_POLICY,
        balancing_policy: BalancingPolicy = DEFAULT_BALANCING_POLICY,
    ) -> None:
        self.state = state if state is not None else SchedulerState()
        self.allocation_engine = AllocationEngine(self.state)
        self.router = LoadBalancingRouter(self.state, policy=balancing_policy)
        self.reclamation_engine = ReclamationEngine(
            self.state, allocation_engine=self.allocation_engine, policy=reclamation_policy
        )
        self._event_seq = itertools.count(1)

        # The most recent "which job" / "which GPU" decision objects,
        # for a decision-trace view (Phase 9) - purely observational,
        # read by nothing in this class. `try_allocate_all` updates
        # both every time it actually commits an assignment; neither
        # is cleared between calls, so a caller can always inspect
        # "the last decision made", not just ones from the current call.
        self.last_allocation_decision: Optional[AllocationDecision] = None
        self.last_routing_decision: Optional[RoutingDecision] = None
        self.last_resource_request_decision: Optional[ReclamationDecision] = None
        #: Every `LoadBalancingTrace` built by the most recent
        #: `_request_additional_gpus_if_needed` call (Day 8) - purely
        #: observational, read by nothing in this class. Replaced (not
        #: appended to) on each call, like the `last_*_decision` fields
        #: above; empty when no waiting job had an outstanding deficit.
        self.last_balancing_traces: List[LoadBalancingTrace] = []
        self._manual_assignment_seq = itertools.count(1)

    # -- setup ----------------------------------------------------------

    def add_user(self, user: User) -> None:
        self.state.add_user(user)

    def add_gpu(self, gpu: GPU) -> None:
        self.allocation_engine.add_gpu(gpu)

    def submit_job(self, job: Job, now: Optional[datetime] = None) -> Event:
        """Register a new job with the Allocation Engine and log that it
        arrived. Does not itself decide anything - see `try_allocate_all`.
        """
        now = now or datetime.now(timezone.utc)
        self.allocation_engine.submit_job(job)
        return self._log_event(
            EventType.REQUEST, now, gpu_id=None, user_id=job.user_id, job_id=job.job_id,
            message=f"{job.job_id} entered the scheduler", reason="job submitted",
        )

    # -- allocation + routing --------------------------------------------

    def try_allocate_all(self, now: Optional[datetime] = None) -> List[AllocationDecision]:
        """Phase 3 picks a job, Phase 5 picks a GPU, commit - repeated
        until no job is waiting or no GPU is routable.

        This is the *only* place "which job" and "which GPU" are
        brought together; neither decision is made here.
        """
        now = now or datetime.now(timezone.utc)
        decisions: List[AllocationDecision] = []
        while True:
            selection = self.allocation_engine.select_next_job(now)
            if selection is None:
                break
            job, policy, reason, candidates = selection

            routing = self.router.route_job(
                job, candidate_gpus=self.allocation_engine.gpu_pool.all_gpus(), now=now
            )
            if routing.outcome is RoutingOutcome.NO_GPU_AVAILABLE:
                break

            gpu = self.state.get_gpu(routing.selected_gpu_id)
            combined_reason = f"{reason} | routed: {routing.reason}"
            decision = self.allocation_engine.finalize_assignment(
                job, gpu, policy, combined_reason, candidates, now=now
            )
            decisions.append(decision)
            self.last_allocation_decision = decision
            self.last_routing_decision = routing

            # A multi-GPU job that just became fully allocated may
            # still have an earlier resource request outstanding for
            # the *same* deficit (raised in a previous pass before
            # this GPU came free) - withdraw it now rather than
            # leaving another user's GPU stuck in IDLE_WARNING asking
            # for capacity nobody needs anymore.
            if job.is_fully_allocated:
                self.reclamation_engine.cancel_pending_requests_for_job(job.job_id, now)

        self._request_additional_gpus_if_needed(now)
        return decisions

    def _holder_priority(self, gpu: GPU) -> Priority:
        holder = self.state.get_user(gpu.assigned_user_id)
        return holder.priority if holder is not None else Priority.LOW

    def _eligible_candidate(self, gpu: GPU, job: Job, now: datetime) -> bool:
        """Shared guard for both reallocation paths below: never a GPU
        already assigned to the requester, already mid-prompt, already
        declined for this exact job, or still in its post-resolution
        cooldown (Phase 12's hysteresis - the actual anti-thrashing
        guard against utilization merely oscillating near a threshold).
        """
        return (
            gpu.is_assigned
            and gpu.assigned_user_id != job.user_id
            and not self.reclamation_engine.has_pending_prompt(gpu.gpu_id)
            and not self.reclamation_engine.has_declined_for(gpu.gpu_id, job.job_id)
            and not self.reclamation_engine.is_in_cooldown(gpu.gpu_id, now, self.router.policy.preemption_cooldown)
        )

    def _shared_skip_reason(self, gpu: GPU, job: Job, now: datetime) -> Optional[str]:
        """Day 8: the same checks `_eligible_candidate` makes, but
        returning *why* a GPU failed instead of only whether it did -
        for `LoadBalancingTrace`, never for the real filtering logic
        above (which stays exactly as it was).
        """
        if not gpu.is_assigned:
            return "not assigned"
        if gpu.assigned_user_id == job.user_id:
            return "held by the requester"
        if self.reclamation_engine.has_pending_prompt(gpu.gpu_id):
            return "prompt already pending"
        if self.reclamation_engine.has_declined_for(gpu.gpu_id, job.job_id):
            return "already declined for this job"
        if self.reclamation_engine.is_in_cooldown(gpu.gpu_id, now, self.router.policy.preemption_cooldown):
            return "in cooldown"
        return None

    def _load_balancing_candidates(
        self, job: Job, now: datetime, path: ReallocationPath, selected_ids: set,
        extra_skip_reason,
    ) -> List[LoadBalancingCandidate]:
        """Day 8: the full, explainable candidate list behind one
        `LoadBalancingTrace` - every GPU in the pool, whether or not it
        was actually asked, with its utilization, holder, cooldown
        status, and (if skipped) why. ``extra_skip_reason(gpu)`` layers
        the path-specific rule (underutilization or priority/score) on
        top of the shared guard above.
        """
        candidates = []
        for gpu in self.state.gpus.values():
            in_cooldown = self.reclamation_engine.is_in_cooldown(gpu.gpu_id, now, self.router.policy.preemption_cooldown)
            if gpu.gpu_id in selected_ids:
                # Selected earlier in this same pass, before the ask it
                # triggered changed its state (e.g. raised its own
                # pending prompt) - it was eligible *at selection time*,
                # which is the only honest thing to report here.
                reason = None
            else:
                reason = self._shared_skip_reason(gpu, job, now)
                if reason is None:
                    reason = extra_skip_reason(gpu)
            candidates.append(LoadBalancingCandidate(
                gpu_id=gpu.gpu_id, utilization_percent=gpu.utilization_percent,
                holder_user_id=gpu.assigned_user_id, eligible=reason is None,
                skip_reason=reason, in_cooldown=in_cooldown, selected=gpu.gpu_id in selected_ids,
            ))
        return candidates

    def _request_additional_gpus_if_needed(self, now: datetime) -> None:
        """Once no genuinely free GPU is left to route (the loop above
        just broke), any still-WAITING job that cannot be fully
        satisfied from the free pool alone gets a chance at two
        further, real sources of capacity - never invented, never
        handed over directly, always through the same confirmation-
        prompt machinery (`ReclamationEngine.request_gpu_for_
        reallocation`) every other reclamation trigger uses:

        1. **Excess-capacity reallocation** (Phase 10) - a multi-GPU
           request (`gpu_count > 1`) still short may ask for another
           user's *underutilized* GPU (below the Tier 2 threshold,
           never an actively-used one - Issue 7/Requirement 4's "do
           not steal an actively-used GPU"). Scoped to `gpu_count > 1`
           on purpose: an ordinary single-GPU job simply waiting its
           turn must never trigger an unsolicited ask just because
           someone else's GPU looks quiet - see `idle_user`/
           `full_lifecycle`'s own scripted timing, which depends on
           that GPU only freeing once its *own* sustained-breach tier
           genuinely completes, not on a bystander's request.

        2. **Priority preemption** (Phase 1 of the 100-scenario fix
           set) - a job of *any* `gpu_count`, including 1, whose
           priority genuinely outranks a current holder's may ask that
           holder to release, even if their GPU is actively busy
           (a real priority gap is judged more urgent than mere
           utilization here - deliberately the opposite eligibility
           rule from path 1, which is why the two are kept separate).
           This still only ever *asks* - the holder's real YES/NO
           answer decides, exactly like every other prompt in this
           project; "immediately preempt" never means bypassing
           confirmation (that is what `Scheduler.force_reclaim`,
           Phase 8, is for).

        What happens to a GPU freed either way - who actually receives
        it - is decided the next time `try_allocate_all` runs, by the
        project's one real allocation policy; this method only asks.
        """
        underutilized_threshold = self.reclamation_engine.policy.tier2.utilization_threshold_percent
        cooldown = self.router.policy.preemption_cooldown
        self.last_balancing_traces = []

        for job in self.state.get_waiting_jobs():
            deficit = job.gpus_still_needed
            if deficit <= 0:
                continue

            requester = self.state.get_user(job.user_id)
            requester_label = requester.name if requester is not None else job.user_id
            requester_priority = requester.priority if requester is not None else Priority.LOW

            asked_gpu_ids: set = set()

            # -- path 1: excess-capacity reallocation (multi-GPU only) --
            if job.gpu_count > 1:
                underutilized_candidates = [
                    gpu for gpu in self.state.gpus.values()
                    if self._eligible_candidate(gpu, job, now)
                    and gpu.utilization_percent < underutilized_threshold
                ]
                underutilized_candidates.sort(
                    key=lambda gpu: (gpu.utilization_percent, self._holder_priority(gpu).value, gpu.gpu_id)
                )
                for gpu in underutilized_candidates[:deficit]:
                    holder_label = self._label_for_holder(gpu)
                    self._log_event(
                        EventType.REQUEST, now, gpu_id=gpu.gpu_id, user_id=job.user_id, job_id=job.job_id,
                        message=(
                            f"{requester_label} needs {deficit} more GPU(s) for {job.job_id}; "
                            f"{gpu.gpu_id} is currently held by {holder_label} but underutilized "
                            f"({gpu.utilization_percent:.1f}% < {underutilized_threshold:.0f}%) - requesting release"
                        ),
                        reason="no free GPU left to cover a multi-GPU request's remaining deficit; "
                               "identified an underutilized candidate instead",
                    )
                    decision = self.reclamation_engine.request_gpu_for_reallocation(
                        gpu, requesting_user_id=job.user_id, requesting_user_name=requester_label,
                        requesting_job_id=job.job_id, now=now,
                    )
                    if decision is not None:
                        self.last_resource_request_decision = decision
                    asked_gpu_ids.add(gpu.gpu_id)

                self.last_balancing_traces.append(LoadBalancingTrace(
                    timestamp=now, job_id=job.job_id, path=ReallocationPath.EXCESS_CAPACITY, deficit=deficit,
                    candidates=self._load_balancing_candidates(
                        job, now, ReallocationPath.EXCESS_CAPACITY, asked_gpu_ids,
                        lambda gpu: None if gpu.utilization_percent < underutilized_threshold else "not underutilized enough",
                    ),
                    selected_gpu_ids=sorted(asked_gpu_ids),
                    reason=f"{job.job_id} needs {deficit} more GPU(s); asked {len(asked_gpu_ids)} underutilized holder(s)",
                ))

            remaining_deficit = deficit - len(asked_gpu_ids)
            if remaining_deficit <= 0:
                continue

            # -- path 2: priority preemption (any gpu_count, including 1) --
            #
            # Eligibility is the *same* allocation score every ordinary
            # waiting-job decision uses - computed head-to-head between
            # the requester and whichever job is actually running on
            # the candidate GPU - not a raw priority-tier comparison.
            # This is deliberate: the project's score formula already
            # has a documented, intentional case (the `ml_video`
            # scenario) where a much smaller MEDIUM-priority job
            # legitimately outscores a much larger HIGH-priority one;
            # a raw "any higher tier always preempts" rule would silently
            # overturn that decision the instant the loser is later
            # marked WAITING, re-litigating a choice the real policy
            # already made. Requiring the *requester's own score* to
            # exceed the *runner's* keeps preemption governed by the
            # one real allocation policy, not a second, competing rule.
            def _preemption_eligible(gpu: GPU) -> bool:
                if not self._eligible_candidate(gpu, job, now):
                    return False
                # A genuine priority-*tier* gap is required in addition
                # to the score comparison below - two jobs of the same
                # priority competing on size/wait time is ordinary
                # allocation policy (Phase 3's territory), never
                # preemption of an already-running job. Without this,
                # any smaller same-priority arrival could "preempt" an
                # existing larger job purely on score, which is not
                # what Phase 1 asks for and would turn routine
                # competition into constant reallocation.
                if self._holder_priority(gpu).value >= requester_priority.value:
                    return False
                holder_job = self.state.get_job(gpu.assigned_job_id) if gpu.assigned_job_id else None
                if holder_job is None:
                    return False
                pair = [job, holder_job]
                return self.allocation_engine.calculate_score(job, pair) > self.allocation_engine.calculate_score(holder_job, pair)

            def _preemption_skip_reason(gpu: GPU) -> Optional[str]:
                if self._holder_priority(gpu).value >= requester_priority.value:
                    return "priority not outranked"
                holder_job = self.state.get_job(gpu.assigned_job_id) if gpu.assigned_job_id else None
                if holder_job is None:
                    return "no running job on this GPU"
                pair = [job, holder_job]
                if self.allocation_engine.calculate_score(job, pair) > self.allocation_engine.calculate_score(holder_job, pair):
                    return None
                return "does not out-score the current holder"

            outranked_candidates = [gpu for gpu in self.state.gpus.values() if gpu.gpu_id not in asked_gpu_ids and _preemption_eligible(gpu)]
            outranked_candidates.sort(key=lambda gpu: (self._holder_priority(gpu).value, gpu.gpu_id))
            preempted_ids: set = set()
            for gpu in outranked_candidates[:remaining_deficit]:
                holder_label = self._label_for_holder(gpu)
                self._log_event(
                    EventType.PRIORITY_PREEMPTION, now, gpu_id=gpu.gpu_id, user_id=job.user_id, job_id=job.job_id,
                    message=(
                        f"{requester_label} ({requester_priority.name}) needs a GPU for {job.job_id}; "
                        f"{gpu.gpu_id} is held by {holder_label} ({self._holder_priority(gpu).name}) - "
                        f"requesting release on priority grounds"
                    ),
                    reason="higher-priority arrival with no free/underutilized GPU available; "
                           "asking a genuinely lower-priority holder instead",
                    metadata={"category": EventType.PRIORITY_PREEMPTION.value},
                )
                decision = self.reclamation_engine.request_gpu_for_reallocation(
                    gpu, requesting_user_id=job.user_id, requesting_user_name=requester_label,
                    requesting_job_id=job.job_id, now=now, is_priority_preemption=True,
                )
                if decision is not None:
                    self.last_resource_request_decision = decision
                preempted_ids.add(gpu.gpu_id)

            self.last_balancing_traces.append(LoadBalancingTrace(
                timestamp=now, job_id=job.job_id, path=ReallocationPath.PRIORITY_PREEMPTION, deficit=remaining_deficit,
                candidates=[
                    c for c in self._load_balancing_candidates(
                        job, now, ReallocationPath.PRIORITY_PREEMPTION, preempted_ids, _preemption_skip_reason,
                    )
                    if c.gpu_id not in asked_gpu_ids
                ],
                selected_gpu_ids=sorted(preempted_ids),
                reason=f"{job.job_id} ({requester_priority.name}) needs {remaining_deficit} more GPU(s); "
                       f"asked {len(preempted_ids)} genuinely lower-priority holder(s)",
            ))

    def _label_for_holder(self, gpu: GPU) -> str:
        holder = self.state.get_user(gpu.assigned_user_id)
        return holder.name if holder is not None else (gpu.assigned_user_id or "unknown")

    # -- manual assignment (demo/test initialization) --------------------

    def manual_assign_gpu(self, gpu_id: str, user_id: str, now: Optional[datetime] = None) -> Job:
        """Admin-only demo/test control: place ``user_id`` directly onto
        a currently-FREE ``gpu_id``, to establish a starting arrangement
        before a demonstration begins (Phase 10's "manual GPU
        assignment / initial test state" - explicitly *not* a
        replacement for the scheduler's own policy: it is a one-time
        setup action, not something the allocation score or FCFS/SJF
        rule ever gets consulted about).

        This still goes through the real commit path
        (`AllocationEngine.finalize_assignment`) rather than poking
        `GPU`/`User`/`Job` fields directly, so a manually-assigned GPU
        is fully indistinguishable, in every index and every later
        reclamation/routing decision, from one the engines assigned
        themselves - it creates one small placeholder `Job` (a
        deliberately long ``estimated_size_minutes`` so it doesn't
        immediately trip an estimated-completion prompt) to be the
        thing that "holds" the GPU, exactly like every other running
        job in this project is the thing that holds its GPU.

        Raises ``KeyError`` for an unknown GPU/user, ``ValueError`` if
        the GPU is not currently free.
        """
        gpu = self.state.get_gpu(gpu_id)
        if gpu is None:
            raise KeyError(f"unknown GPU {gpu_id!r}")
        user = self.state.get_user(user_id)
        if user is None:
            raise KeyError(f"unknown user {user_id!r}")
        if gpu.is_assigned:
            raise ValueError(f"{gpu_id} is not free (already assigned to {gpu.assigned_user_id!r})")

        now = now or datetime.now(timezone.utc)
        job = Job(
            job_id=f"MANUAL-{next(self._manual_assignment_seq)}", user_id=user_id, name="Manual Assignment",
            priority=user.priority, estimated_size_minutes=100_000, submitted_at=now,
        )
        self.submit_job(job, now=now)
        candidates = [CandidateInfo(
            job_id=job.job_id, user_id=user_id, priority=job.priority,
            size_minutes=job.estimated_size_minutes, waiting_time=timedelta(0), score=None,
        )]
        decision = self.allocation_engine.finalize_assignment(
            job, gpu, AllocationPolicy.FCFS, "manual admin assignment (demo/test initialization)",
            candidates, now=now,
        )
        self.last_allocation_decision = decision
        return job

    # -- utilization / reclamation ---------------------------------------

    def record_utilization(
        self, gpu_id: str, utilization_percent: float, timestamp: datetime,
        memory_used_mb: Optional[float] = None,
    ) -> Optional[ReclamationDecision]:
        gpu = self.state.get_gpu(gpu_id)
        if gpu is None:
            raise ValueError(f"unknown GPU {gpu_id!r}")
        observation = UtilizationObservation(
            utilization_percent=utilization_percent, timestamp=timestamp, memory_used_mb=memory_used_mb,
        )
        return self.reclamation_engine.record_utilization(gpu, observation)

    def respond_to_prompt(
        self, gpu_id: str, response: ConfirmationResponse, now: Optional[datetime] = None
    ) -> ReclamationDecision:
        return self.reclamation_engine.respond(gpu_id, response, now=now)

    def check_reclamation_timeouts(self, now: datetime) -> List[ReclamationDecision]:
        return self.reclamation_engine.check_timeouts(now)

    def check_estimated_completions(self, now: datetime) -> List[ReclamationDecision]:
        """For every RUNNING job whose estimated duration has elapsed,
        ask (through the real confirmation flow) whether the GPU is
        still needed - never silently mark the job complete.

        Reaching an *estimated* completion time is not proof a real
        workload actually finished (Part 35's own caution) - it is
        exactly the same kind of "should we check in?" signal a
        sustained-low-utilization tier is, so it is routed through
        `ReclamationEngine.prompt_for_job_completion`, the same
        watch/prompt/respond/timeout machinery every other prompt in
        this project already uses. A GPU that already has a prompt
        pending (e.g. a utilization tier got there first) is left
        alone rather than double-prompted.
        """
        decisions: List[ReclamationDecision] = []
        for job in list(self.state.jobs.values()):
            if job.status != JobStatus.RUNNING or job.started_at is None or not job.assigned_gpu_ids:
                continue
            estimated_completion = job.started_at + timedelta(minutes=job.estimated_size_minutes)
            if now < estimated_completion:
                continue
            # A multi-GPU job's estimated duration is for the whole
            # request, not any one GPU - every GPU it currently holds
            # gets its own prompt (the same confirmation each GPU in
            # this project always gets), not just the first.
            for gpu_id in list(job.assigned_gpu_ids):
                gpu = self.state.get_gpu(gpu_id)
                if gpu is None:
                    continue
                decision = self.reclamation_engine.prompt_for_job_completion(gpu, now)
                if decision is not None:
                    decisions.append(decision)
        return decisions

    # -- completion ----------------------------------------------------------

    def complete_job(self, job_id: str, now: Optional[datetime] = None) -> Event:
        """A job finishes normally (not reclaimed) - releases its GPU
        the same way a reclaim does, but as a distinct `JOB_COMPLETION`
        event rather than a `RECLAIM` one (Phase 5 of the 100-scenario
        fix set - completion and reclaim are different things, and now
        say so in the event log itself).
        """
        now = now or datetime.now(timezone.utc)
        job = self.state.get_job(job_id)
        if job is None:
            raise ValueError(f"unknown job {job_id!r}")
        if job.status != JobStatus.RUNNING:
            raise ValueError(f"job {job_id!r} is not RUNNING (status={job.status.value})")

        gpu_ids = list(job.assigned_gpu_ids)
        user = self.state.get_user(job.user_id)

        job.status = JobStatus.COMPLETED
        job.assigned_gpu_ids.clear()

        for gpu_id in gpu_ids:
            gpu = self.state.get_gpu(gpu_id)
            if gpu is None:
                continue
            # A prompt outstanding on this GPU (any tier, or a
            # resource-request/preemption ask) was asking about *this*
            # job/holder - it is now moot, and must never be left to
            # later auto-reclaim (or be answered into reclaiming)
            # whoever the GPU is reassigned to next. Found via a real
            # regression while testing Phase 1's preemption path
            # against the `ml_video` scenario.
            self.reclamation_engine.clear_watch_for_gpu(gpu_id, now)
            assignment = self.state.get_active_assignment_for_gpu(gpu.gpu_id)
            if assignment is not None:
                assignment.end(now)
            gpu.assigned_user_id = None
            gpu.assigned_job_id = None
            gpu.status = GPUStatus.IDLE
            if user is not None and gpu.gpu_id in user.assigned_gpu_ids:
                user.assigned_gpu_ids.remove(gpu.gpu_id)
            # Day 4 (user/job management audit): every other release
            # path (`force_reclaim`, `handle_gpu_failure`,
            # `ReclamationEngine._reclaim`) already calls this to keep
            # the HashMap-backed `UserGPUIndex` in sync - a normal
            # completion was the one release path that didn't, so
            # `AllocationEngine.get_gpus_for_user` kept reporting a GPU
            # the user no longer held (SchedulerState/User.
            # assigned_gpu_ids were already correct; only this index
            # was stale). The index is never a second source of truth,
            # only an efficient view over the one real state - this
            # keeps it that way for every release path, not just some.
            self.allocation_engine.release_user_gpu(job.user_id, gpu_id)

        if user is not None and job_id in user.running_job_ids:
            user.running_job_ids.remove(job_id)

        message = f"{job_id} completed" + (f", released {', '.join(gpu_ids)}" if gpu_ids else "")
        event = self._log_event(
            EventType.JOB_COMPLETION, now, gpu_id=(gpu_ids[0] if gpu_ids else None), user_id=job.user_id, job_id=job_id,
            message=message, reason="job completed normally",
        )

        for gpu_id in gpu_ids:
            self.allocation_engine.mark_gpu_available(gpu_id)

        return event

    # -- waiting-job cancellation (Phase 7) --------------------------------

    def cancel_job(self, job_id: str, now: Optional[datetime] = None) -> Event:
        """Withdraw a still-WAITING job before it was ever assigned a
        GPU (100-scenario validation, Phase 7). Never touches a
        RUNNING job - that has real resources to release, which is
        `complete_job` (finished) or a reclaim's job, not a
        cancellation; raises ``ValueError`` rather than silently doing
        the wrong thing if called on one.

        Removes the job from `AllocationEngine`'s own FIFO (so it can
        never be selected by a later `select_next_job`), marks it
        `JobStatus.CANCELLED`, and logs `JOB_CANCELLED` - the one
        user-facing signal a portal needs to show "your request was
        cancelled" rather than leaving it looking like it's still
        pending forever.
        """
        now = now or datetime.now(timezone.utc)
        job = self.state.get_job(job_id)
        if job is None:
            raise ValueError(f"unknown job {job_id!r}")
        if job.status == JobStatus.RUNNING:
            raise ValueError(
                f"job {job_id!r} is RUNNING - use complete_job/respond_to_prompt/force_reclaim, "
                f"not cancel_job, for a job that already holds a GPU"
            )
        if job.status != JobStatus.WAITING:
            raise ValueError(f"job {job_id!r} is not WAITING (status={job.status.value}) - nothing to cancel")

        self.allocation_engine.remove_waiting_job(job_id)
        job.status = JobStatus.CANCELLED

        return self._log_event(
            EventType.JOB_CANCELLED, now, gpu_id=None, user_id=job.user_id, job_id=job_id,
            message=f"{job_id} cancelled by {job.user_id} while still waiting",
            reason="user withdrew a still-waiting request",
        )

    # -- admin force-reclaim (Phase 8) -------------------------------------

    def force_reclaim(self, gpu_id: str, now: Optional[datetime] = None) -> Event:
        """Admin-only bypass of the normal confirmation flow (100-
        scenario validation, Phase 8) - the one place in this project
        a GPU is taken back *without* asking its holder first. Every
        other reclamation trigger (a sustained breach, a priority
        preemption ask, a resource request) always goes through
        `ReclamationEngine.request_gpu_for_reallocation`/`_raise_prompt`
        and waits for a real YES/NO; this is deliberately the one
        exception, reserved for an admin action, never invoked by
        ordinary scheduling logic.

        Clears any prompt that happened to be pending on ``gpu_id``
        first (an admin override should never leave a stale watch
        behind - see `complete_job`'s own fix for exactly that class
        of bug), releases the GPU and its holder's bookkeeping the
        same way any other reclaim does, logs `ADMIN_FORCE_RECLAIM`,
        and reevaluates the waiting queue so the freed capacity is
        used immediately, through the ordinary allocation policy.
        """
        now = now or datetime.now(timezone.utc)
        gpu = self.state.get_gpu(gpu_id)
        if gpu is None:
            raise KeyError(f"unknown GPU {gpu_id!r}")
        if not gpu.is_assigned:
            raise ValueError(f"{gpu_id} is not currently assigned to anyone")

        self.reclamation_engine.clear_watch_for_gpu(gpu_id, now)

        job = self.state.get_job(gpu.assigned_job_id) if gpu.assigned_job_id else None
        user_id = gpu.assigned_user_id
        user = self.state.get_user(user_id) if user_id else None
        user_label = user.name if user is not None else user_id

        if job is not None:
            if gpu_id in job.assigned_gpu_ids:
                job.assigned_gpu_ids.remove(gpu_id)
            if not job.assigned_gpu_ids:
                job.status = JobStatus.RECLAIMED
        if user is not None:
            if gpu_id in user.assigned_gpu_ids:
                user.assigned_gpu_ids.remove(gpu_id)
            if job is not None and job.job_id in user.running_job_ids:
                user.running_job_ids.remove(job.job_id)

        assignment = self.state.get_active_assignment_for_gpu(gpu_id)
        if assignment is not None:
            assignment.end(now)

        gpu.assigned_user_id = None
        gpu.assigned_job_id = None
        gpu.status = GPUStatus.IDLE
        self.allocation_engine.mark_gpu_available(gpu_id)
        if user_id is not None:
            self.allocation_engine.release_user_gpu(user_id, gpu_id)

        event = self._log_event(
            EventType.ADMIN_FORCE_RECLAIM, now, gpu_id=gpu_id, user_id=user_id,
            job_id=job.job_id if job is not None else None,
            message=f"Admin force-reclaimed {gpu_id} from {user_label}",
            reason="admin bypassed the normal confirmation flow",
            metadata={"category": EventType.ADMIN_FORCE_RECLAIM.value},
        )
        self.try_allocate_all(now=now)
        return event

    # -- priority mutation (Phase 9) ---------------------------------------

    def change_job_priority(self, job_id: str, new_priority: Priority, now: Optional[datetime] = None) -> Event:
        """Admin-only: change a still-WAITING job's priority (100-
        scenario validation, Phase 9) - e.g. LOW -> HIGH. Never
        touches a RUNNING job's priority (it already has its GPU;
        priority no longer decides anything for it).

        The job object *is* what `AllocationEngine.select_next_job`
        reads on its next call - there is no separate priority-queue
        node to "reinsert"; changing `Job.priority` in place and
        letting the next `try_allocate_all` recompute from scratch is
        the actual, correct re-insertion (this project's waiting
        structures were never a snapshot of a job's priority, they
        always read it live).
        """
        now = now or datetime.now(timezone.utc)
        job = self.state.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id!r}")
        if job.status != JobStatus.WAITING:
            raise ValueError(f"job {job_id!r} is not WAITING (status={job.status.value})")

        old_priority = job.priority
        job.priority = new_priority

        event = self._log_event(
            EventType.PRIORITY_CHANGED, now, gpu_id=None, user_id=job.user_id, job_id=job_id,
            message=f"{job_id} priority changed: {old_priority.name} -> {new_priority.name}",
            reason="admin priority mutation",
            metadata={"category": EventType.PRIORITY_CHANGED.value, "old_priority": old_priority.name, "new_priority": new_priority.name},
        )
        self.try_allocate_all(now=now)
        return event

    # -- maintenance mode (Phase 14) ---------------------------------------

    def set_gpu_maintenance(self, gpu_id: str, now: Optional[datetime] = None) -> Event:
        """Admin-only: take a currently-FREE GPU out of the allocatable
        pool without deleting it (100-scenario validation, Phase 14).
        Raises ``ValueError`` if the GPU is currently assigned - an
        admin must resolve that assignment (let it finish, or
        `force_reclaim` it) before parking the GPU; maintenance is
        never used to silently interrupt a running job.
        """
        now = now or datetime.now(timezone.utc)
        gpu = self.state.get_gpu(gpu_id)
        if gpu is None:
            raise KeyError(f"unknown GPU {gpu_id!r}")
        if gpu.is_assigned:
            raise ValueError(f"{gpu_id} is currently assigned - resolve its assignment before entering maintenance")
        gpu.status = GPUStatus.MAINTENANCE
        return self._log_event(
            EventType.SYSTEM, now, gpu_id=gpu_id, user_id=None, job_id=None,
            message=f"{gpu_id} entered MAINTENANCE - removed from the allocatable pool",
            reason="admin maintenance action",
        )

    def clear_gpu_maintenance(self, gpu_id: str, now: Optional[datetime] = None) -> Event:
        """Restore a GPU from MAINTENANCE back to IDLE, and reevaluate
        the waiting queue - the newly-available capacity is used
        immediately through the ordinary allocation policy, never
        handed directly to whichever job happens to be first."""
        now = now or datetime.now(timezone.utc)
        gpu = self.state.get_gpu(gpu_id)
        if gpu is None:
            raise KeyError(f"unknown GPU {gpu_id!r}")
        if gpu.status != GPUStatus.MAINTENANCE:
            raise ValueError(f"{gpu_id} is not in MAINTENANCE (status={gpu.status.value})")
        gpu.status = GPUStatus.IDLE
        self.allocation_engine.mark_gpu_available(gpu_id)
        event = self._log_event(
            EventType.SYSTEM, now, gpu_id=gpu_id, user_id=None, job_id=None,
            message=f"{gpu_id} left MAINTENANCE - available again",
            reason="admin maintenance action",
        )
        self.try_allocate_all(now=now)
        return event

    # -- hardware failure handling (Phase 6) -------------------------------

    def handle_gpu_failure(self, gpu_id: str, now: Optional[datetime] = None) -> Optional[Event]:
        """The hardware layer reported ``gpu_id`` as gone/errored - it
        vanished from a poll, or NVML/nvidia-smi reported an error for
        it specifically (100-scenario validation, Phase 6). Distinct
        from every other release path: nobody asked, nothing timed
        out, this is the hardware itself failing.

        Marks the GPU `UNAVAILABLE` (kept, never deleted - the object
        and its history survive, matching `set_gpu_maintenance`'s own
        "never delete a GPU" rule), clears any stale watch, releases
        whoever was assigned (their job goes back to WAITING with
        this GPU removed from its `assigned_gpu_ids` - a multi-GPU job
        keeps its other GPUs, exactly like an ordinary partial
        reclaim), logs `HARDWARE_FAILURE`, and reevaluates the waiting
        queue so any *other* free GPU is used immediately. Returns
        ``None`` if the GPU is already unknown or already
        UNAVAILABLE - failing twice is a no-op, not an error.
        """
        now = now or datetime.now(timezone.utc)
        gpu = self.state.get_gpu(gpu_id)
        if gpu is None or gpu.status == GPUStatus.UNAVAILABLE:
            return None

        self.reclamation_engine.clear_watch_for_gpu(gpu_id, now)

        job = self.state.get_job(gpu.assigned_job_id) if gpu.assigned_job_id else None
        user_id = gpu.assigned_user_id
        user = self.state.get_user(user_id) if user_id else None
        user_label = user.name if user is not None else user_id

        if job is not None:
            if gpu_id in job.assigned_gpu_ids:
                job.assigned_gpu_ids.remove(gpu_id)
            # A multi-GPU job that still holds other GPUs keeps
            # running on them; only a job left with none goes back to
            # WAITING - it still needs replacement capacity, and stays
            # a normal, real candidate for the next allocation round,
            # never silently dropped.
            if not job.assigned_gpu_ids:
                job.status = JobStatus.WAITING
                self.allocation_engine.requeue_job(job)
        if user is not None:
            if gpu_id in user.assigned_gpu_ids:
                user.assigned_gpu_ids.remove(gpu_id)
            if job is not None and not job.assigned_gpu_ids and job.job_id in user.running_job_ids:
                user.running_job_ids.remove(job.job_id)

        assignment = self.state.get_active_assignment_for_gpu(gpu_id)
        if assignment is not None:
            assignment.end(now)

        gpu.assigned_user_id = None
        gpu.assigned_job_id = None
        gpu.status = GPUStatus.UNAVAILABLE
        # Kept in SchedulerState/GPUPool, never removed - the object
        # and its history survive (Phase 14's own "never delete a GPU"
        # rule, applied here too); `is_gpu_available` already excludes
        # anything that isn't plainly IDLE, so UNAVAILABLE is
        # automatically never routed to without needing to pop it out
        # of the pool (which would leave exactly the dangling User/Job
        # reference `remove_gpu` is documented to risk).
        if user_id is not None:
            self.allocation_engine.release_user_gpu(user_id, gpu_id)

        event = self._log_event(
            EventType.HARDWARE_FAILURE, now, gpu_id=gpu_id, user_id=user_id,
            job_id=job.job_id if job is not None else None,
            message=f"{gpu_id} reported UNAVAILABLE by the hardware layer" + (f" - {user_label} affected" if user_label else ""),
            reason="hardware poll reported this GPU as gone/errored",
            metadata={"category": EventType.HARDWARE_FAILURE.value},
        )
        self.try_allocate_all(now=now)
        return event

    def _log_event(
        self, event_type: EventType, now: datetime, gpu_id: Optional[str],
        user_id: Optional[str], job_id: Optional[str], message: str, reason: str,
        metadata: Optional[dict] = None,
    ) -> Event:
        event = Event(
            event_id=f"S{next(self._event_seq)}", event_type=event_type, timestamp=now,
            gpu_id=gpu_id, user_id=user_id, job_id=job_id, message=message, reason=reason,
            metadata=metadata or {},
        )
        self.state.log_event(event)
        return event
