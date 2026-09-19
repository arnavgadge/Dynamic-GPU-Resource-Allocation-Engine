"""The Allocation Engine: decides which waiting job gets a free GPU.

This is the first phase allowed to make a decision. It answers
exactly one question - "a GPU is available, several jobs are
waiting, which job gets it?" - by following the project's fixed
policy:

    GPU becomes available
            v
    Check waiting jobs
            v
    Check critical jobs
            v
    Calculate allocation information
            v
    Check job-size similarity
            v
    Choose FCFS or score-based/SJF behavior
            v
    Select winning job
            v
    Assign GPU
            v
    Update system state
            v
    Generate allocation event

It reuses every Phase 2 structure for its stated purpose - a
`GPUPool` (linked list) for the full GPU inventory, a
`GPUUtilizationHeap` (min-heap) restricted to only the GPUs that are
currently *available* (never confused with "least utilized among
all GPUs" - see `mark_gpu_available`), a `WaitingJobQueue` (FIFO) as
the actual arrival-ordered source of truth for waiting jobs, a
`UserGPUIndex` (hashmap) for O(1) user -> GPU lookups, and - for the
score-based branch specifically - a `PriorityQueue` (a Max-Heap under
the hood) to actually pick the highest-scoring job, rather than a
plain Python `min()`/`max()` standing in for it. Nothing here
implements GPU reclamation, load balancing, or leases.
"""

import itertools
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from engine.allocation.config import JOB_SIZE_SIMILARITY_THRESHOLD
from engine.allocation.decision import AllocationDecision, AllocationPolicy, CandidateInfo
from engine.allocation.scoring import ScoreBreakdown, allocation_score, calculate_allocation_score
from engine.balancing.availability import is_gpu_available
from engine.allocation.similarity import relative_size_spread
from engine.dsa.gpu_pool import GPUPool
from engine.dsa.gpu_utilization_heap import GPUUtilizationHeap
from engine.dsa.priority_queue import PriorityQueue
from engine.dsa.user_gpu_index import UserGPUIndex
from engine.dsa.waiting_job_queue import WaitingJobQueue
from engine.models.assignment import GPUAssignment
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState

# The result of *deciding* a winner, before anything is mutated:
# (winning job, policy used, human-readable reason, every candidate considered).
_Selection = Tuple[Job, AllocationPolicy, str, List[CandidateInfo]]


class AllocationEngine:
    """Allocates available GPUs to waiting jobs under the project's policy.

    Owns no data the `SchedulerState` doesn't also hold - it mirrors
    GPUs/jobs into the DSA structures that make each specific
    question (least-utilized *available* GPU, longest-waiting job)
    fast to answer, and keeps `SchedulerState` as the record of
    truth for anything a later phase or the frontend needs to read.
    """

    def __init__(self, state: SchedulerState) -> None:
        self.state = state

        self._pool = GPUPool()
        self._available_gpus = GPUUtilizationHeap()
        self._waiting_jobs: WaitingJobQueue = WaitingJobQueue()
        self._user_index = UserGPUIndex()

        # GPUs removed from the pool may still be sitting in
        # `_available_gpus` (a min-heap has no O(log n) arbitrary
        # delete) - this lets extraction skip them lazily instead of
        # handing out a GPU that no longer exists. See `_pop_available_gpu`.
        self._removed_gpu_ids: set = set()

        self._assignment_seq = itertools.count(1)
        self._event_seq = itertools.count(1)

    # -- GPU pool management -----------------------------------------

    def add_gpu(self, gpu: GPU) -> None:
        """Add a GPU to the company pool. If it's already free, it is
        immediately available for allocation."""
        self.state.add_gpu(gpu)
        self._pool.add_gpu(gpu)
        if not gpu.is_assigned:
            self._available_gpus.insert_gpu(gpu)

    def remove_gpu(self, gpu_id: str) -> bool:
        """Remove a GPU from the pool (e.g. decommissioned). O(n) - see `GPUPool.remove_gpu`."""
        removed = self._pool.remove_gpu(gpu_id)
        if removed:
            self._removed_gpu_ids.add(gpu_id)
            self.state.gpus.pop(gpu_id, None)
        return removed

    def mark_gpu_available(self, gpu_id: str) -> bool:
        """Signal that an already-known, currently-unassigned GPU is free
        to be allocated. Returns False if the GPU is unknown or still
        assigned to someone - this method only ever adds *available*
        GPUs to the min-heap, it never uses utilization as a stand-in
        for availability.
        """
        gpu = self._pool.get_gpu(gpu_id)
        if gpu is None or gpu.is_assigned:
            return False
        self._available_gpus.insert_gpu(gpu)
        return True

    def _pop_available_gpu(self) -> Optional[GPU]:
        """Take the least-utilized currently-available GPU, if any.

        Lazily discards stale heap entries (a GPU that was removed
        from the pool, or - defensively - one that somehow got
        assigned without going through this engine) instead of
        handing out something invalid.
        """
        while not self._available_gpus.is_empty():
            gpu = self._available_gpus.extract_least_utilized()
            if gpu is None:
                break
            if gpu.gpu_id in self._removed_gpu_ids or gpu.is_assigned:
                continue
            return gpu
        return None

    # -- Job submission -------------------------------------------------

    def submit_job(self, job: Job) -> None:
        """Register a new job and, if it is waiting, enter it into the
        FIFO waiting queue in true arrival order.

        Rejects a duplicate ``job_id`` and a job that isn't
        ``WAITING`` (e.g. already ``RUNNING``) - both are submission
        mistakes the engine should fail on clearly rather than
        silently accept.
        """
        if job.job_id in self.state.jobs:
            raise ValueError(f"job {job.job_id!r} has already been submitted")
        if job.status != JobStatus.WAITING:
            raise ValueError(
                f"job {job.job_id!r} must be WAITING to be submitted, got {job.status.value}"
            )
        self.state.add_job(job)
        self._waiting_jobs.enqueue_job(job)

    def _current_waiting_jobs(self) -> List[Job]:
        """Every job presently in the FIFO, in true arrival order.

        Filters defensively for `JobStatus.WAITING` in case a job's
        status changed through some path other than this engine's
        own assignment logic.
        """
        return [job for job in self._waiting_jobs.to_list() if job.status == JobStatus.WAITING]

    def _remove_from_waiting_queue(self, job_id: str) -> None:
        """Remove one job from the FIFO, keeping everyone else's relative order.

        `WaitingJobQueue`/`Queue` only expose front-removal, so an
        interior removal is done the same way `MinHeap.rebuild` fixes
        a stale heap: drain everything and re-enqueue what should
        remain. O(n) in the number of currently-waiting jobs.
        """
        remaining = [job for job in self._waiting_jobs.to_list() if job.job_id != job_id]
        self._waiting_jobs = WaitingJobQueue()
        for job in remaining:
            self._waiting_jobs.enqueue_job(job)

    def requeue_job(self, job: Job) -> None:
        """Put an already-known job back into the FIFO as a genuine,
        live waiting candidate (Phase 6's hardware-failure handling -
        a job that just lost its GPU to a hardware failure, not a
        fresh submission). Unlike `submit_job`, this never checks "is
        this job_id new" (it isn't) - the caller is responsible for
        having already set `job.status = JobStatus.WAITING` and
        cleared whatever GPUs it no longer holds; this only makes sure
        `select_next_job` can actually see it again.

        Idempotent (Day 5): a partially-allocated multi-GPU job is
        already WAITING *and* still in the FIFO; if it then loses its
        last held GPU, the caller requeues it - which used to enqueue
        a second copy, so the same job competed against itself and
        `waiting_job_ids()` reported it twice. A job already present
        is simply left where it is (its original arrival position).
        """
        if any(queued.job_id == job.job_id for queued in self._waiting_jobs.to_list()):
            return
        self._waiting_jobs.enqueue_job(job)

    def remove_waiting_job(self, job_id: str) -> None:
        """Public wrapper over `_remove_from_waiting_queue` - for a
        caller outside this engine that needs a WAITING job gone from
        the FIFO without going through the normal assignment path
        (`Scheduler.cancel_job`, Phase 7's withdrawal). Does not touch
        `Job.status` itself - the caller decides what status the
        cancelled job actually ends up in.
        """
        self._remove_from_waiting_queue(job_id)

    # -- Candidate selection ------------------------------------------

    def select_next_job(self, now: Optional[datetime] = None) -> Optional[_Selection]:
        """Decide which waiting job would win the next available GPU.

        Pure decision logic - reads `SchedulerState`/the waiting
        queue but changes nothing. Returns ``None`` if no job is
        waiting. ``now`` is only used by the score-based branch, to
        compute each candidate's starvation-prevention aging bonus
        (Phase 2 of the 100-scenario fix set) - omitting it is
        equivalent to "no time has passed" (zero aging for everyone),
        never an error.
        """
        candidates = self._current_waiting_jobs()
        if not candidates:
            return None

        critical_candidates = [job for job in candidates if job.priority == Priority.CRITICAL]
        pool = critical_candidates if critical_candidates else candidates
        tier_note = (
            f"{len(critical_candidates)} critical job(s) waiting - scoring restricted to critical jobs"
            if critical_candidates
            else "no critical jobs waiting - scoring open to all waiting jobs"
        )

        if len(pool) == 1:
            winner = pool[0]
            reason = f"{tier_note}; only one eligible candidate ({winner.job_id}) - trivially FCFS"
            return winner, AllocationPolicy.FCFS, reason, [self._candidate_info(winner, breakdown=None)]

        sizes = [job.estimated_size_minutes for job in pool]
        spread = relative_size_spread(sizes)
        threshold_pct = f"{JOB_SIZE_SIMILARITY_THRESHOLD:.0%}"

        if spread <= JOB_SIZE_SIMILARITY_THRESHOLD:
            return self._select_fcfs(pool, tier_note, spread, threshold_pct)
        return self._select_score_based(pool, tier_note, spread, threshold_pct, now)

    def _select_fcfs(self, pool: List[Job], tier_note: str, spread: float, threshold_pct: str) -> _Selection:
        # The waiting queue's FIFO order matches submission order in
        # the normal (real-time) case, but the winner is picked from
        # `Job.submitted_at` directly rather than trusting queue
        # position - that is the "actual submission information" the
        # policy is defined on, and it stays correct even if jobs
        # were ever loaded/replayed out of call-order (bulk import,
        # tests, a restarted engine reading persisted jobs back in).
        winner = min(pool, key=lambda job: job.submitted_at)
        reason = (
            f"{tier_note}; job sizes within {threshold_pct} of each other "
            f"(spread={spread:.1%}) -> FCFS; {winner.job_id} has waited longest"
        )
        candidates = [self._candidate_info(job, breakdown=None) for job in pool]
        return winner, AllocationPolicy.FCFS, reason, candidates

    def _select_score_based(
        self, pool: List[Job], tier_note: str, spread: float, threshold_pct: str, now: Optional[datetime] = None,
    ) -> _Selection:
        sizes = [job.estimated_size_minutes for job in pool]
        min_size, max_size = min(sizes), max(sizes)
        breakdowns = {job.job_id: calculate_allocation_score(job, min_size, max_size, now) for job in pool}

        # The winner is actually decided by the Phase 2 Priority Queue
        # (a Max-Heap under the hood) - not a plain Python min()/max()
        # - so "Priority Queue manages waiting jobs, Max Heap finds the
        # highest-value one" is what genuinely runs, not just what the
        # project's DSA write-up claims. The key is a tuple so ties are
        # broken by earliest submission (the same "whoever has waited
        # longest" idea FCFS uses) without relying on the heap's own
        # insertion-order tie-break, which is about call order and can
        # diverge from `Job.submitted_at` (see `_select_fcfs`'s own
        # docstring for why that distinction matters). The aging
        # component already folded into each breakdown's `final_score`
        # (Phase 2) is what actually lets a long-waiting job win here
        # despite a lower base score - see `calculate_aging_component`.
        score_queue: PriorityQueue[Job] = PriorityQueue(
            key_fn=lambda job: (breakdowns[job.job_id].final_score, -job.submitted_at.timestamp())
        )
        for job in pool:
            score_queue.insert(job)
        winner = score_queue.pop_best()
        winner_breakdown = breakdowns[winner.job_id]

        reason = (
            f"{tier_note}; job sizes differ beyond {threshold_pct} "
            f"(spread={spread:.1%}) -> score-based selection; "
            f"{winner.job_id} scored highest ({winner_breakdown.final_score:.3f}"
            f"{f', incl. +{winner_breakdown.aging_component:.3f} aging' if winner_breakdown.aging_component > 0 else ''})"
        )
        candidates = [self._candidate_info(job, breakdown=breakdowns[job.job_id]) for job in pool]
        return winner, AllocationPolicy.SCORE_BASED, reason, candidates

    def _candidate_info(self, job: Job, breakdown: Optional[ScoreBreakdown]) -> CandidateInfo:
        return CandidateInfo(
            job_id=job.job_id,
            user_id=job.user_id,
            priority=job.priority,
            size_minutes=job.estimated_size_minutes,
            waiting_time=job.waiting_time,
            score=breakdown.final_score if breakdown is not None else None,
            base_score=breakdown.base_score if breakdown is not None else None,
            aging_component=breakdown.aging_component if breakdown is not None else None,
            waiting_minutes=breakdown.waiting_minutes if breakdown is not None else None,
            priority_component=breakdown.priority_component if breakdown is not None else None,
            size_component=breakdown.size_component if breakdown is not None else None,
        )

    def calculate_score(self, job: Job, candidates: List[Job], now: Optional[datetime] = None) -> float:
        """The project's allocation score for ``job``, relative to
        ``candidates`` (the set it is actually competing against).
        Exposed directly for tests/inspection outside a full decision.
        Includes the aging bonus when ``now`` is given; omitting ``now``
        is equivalent to zero elapsed wait, same as `select_next_job`.
        """
        sizes = [candidate.estimated_size_minutes for candidate in candidates]
        return allocation_score(job, min(sizes), max(sizes), now)

    # -- Assignment ------------------------------------------------------

    def _assign_gpu(
        self, job: Job, gpu: GPU, policy: AllocationPolicy, reason: str, candidates: List[CandidateInfo],
        now: Optional[datetime] = None,
    ) -> AllocationDecision:
        """Commit a decision: move the job to RUNNING, the GPU to ACTIVE,
        record the assignment, update every lookup structure, and log
        the allocation event. See the module/README for the consistency
        this keeps between GPU/Job/User/UserGPUIndex.

        ``now`` defaults to the real wall clock only when the caller
        doesn't supply one - a simulated caller (Phase 6) always passes
        its own simulated ``now`` so `Job.started_at`, the
        `GPUAssignment`, and the logged events all carry simulated
        rather than real timestamps.
        """
        now = now or datetime.now(timezone.utc)

        # One GPU is committed to `job` per call, exactly as before -
        # what changes for a multi-GPU request (`job.gpu_count > 1`)
        # is that this may still leave the job short: it only leaves
        # the waiting queue and becomes RUNNING once it holds
        # `gpu_count` GPUs. Until then it stays WAITING (and stays in
        # the queue) so the *next* call to `select_next_job` can offer
        # it - or a higher-priority/earlier job - the next available
        # GPU, under the exact same policy every other decision uses.
        if gpu.gpu_id not in job.assigned_gpu_ids:
            job.assigned_gpu_ids.append(gpu.gpu_id)
        if job.started_at is None:
            job.started_at = now

        gpu.assigned_user_id = job.user_id
        gpu.assigned_job_id = job.job_id
        gpu.status = GPUStatus.ACTIVE

        user = self.state.get_user(job.user_id)
        if user is not None:
            if gpu.gpu_id not in user.assigned_gpu_ids:
                user.assigned_gpu_ids.append(gpu.gpu_id)
            if job.job_id not in user.running_job_ids:
                user.running_job_ids.append(job.job_id)

        if job.is_fully_allocated:
            job.status = JobStatus.RUNNING
            self._remove_from_waiting_queue(job.job_id)

        assignment = GPUAssignment(
            assignment_id=f"A{next(self._assignment_seq)}",
            gpu_id=gpu.gpu_id,
            user_id=job.user_id,
            job_id=job.job_id,
            created_at=now,
        )
        self.state.add_assignment(assignment)
        self._user_index.add_assignment(assignment)

        user_label = user.name if user is not None else job.user_id
        # `gpu_count == 1` (the default, and every job before this
        # phase) keeps the exact original single-GPU message; a
        # multi-GPU request additionally reports progress toward the
        # full request, e.g. "(2/3 GPUs)", so the event log/decision
        # trace shows partial satisfaction rather than looking like a
        # single, complete assignment.
        progress = f" ({len(job.assigned_gpu_ids)}/{job.gpu_count} GPUs)" if job.gpu_count > 1 else ""
        event = Event(
            event_id=f"E{next(self._event_seq)}",
            event_type=EventType.ALLOC,
            timestamp=now,
            gpu_id=gpu.gpu_id,
            user_id=job.user_id,
            job_id=job.job_id,
            message=f"Assigned {gpu.gpu_id} to {user_label}{progress}",
            reason=reason,
            metadata={"policy": policy.value},
        )
        self.state.log_event(event)

        # A separate STATUS event, in addition to ALLOC - the same
        # pattern Phase 4's confirmation flow already uses when a GPU
        # backs off to ACTIVE. `decision.event` below still points at
        # the ALLOC event (the one every existing caller/test reads);
        # this one exists purely for a complete, readable event log.
        self.state.log_event(Event(
            event_id=f"E{next(self._event_seq)}",
            event_type=EventType.STATUS,
            timestamp=now,
            gpu_id=gpu.gpu_id,
            user_id=job.user_id,
            job_id=job.job_id,
            message=f"{gpu.gpu_id} transitioned to ACTIVE",
            reason=reason,
        ))

        return AllocationDecision(
            timestamp=now,
            gpu_id=gpu.gpu_id,
            job_id=job.job_id,
            user_id=job.user_id,
            policy=policy,
            reason=reason,
            candidates=candidates,
            event=event,
        )

    def finalize_assignment(
        self, job: Job, gpu: GPU, policy: AllocationPolicy, reason: str, candidates: List[CandidateInfo],
        now: Optional[datetime] = None,
    ) -> AllocationDecision:
        """Commit a (job, GPU) pairing chosen by an external router
        (Phase 5's `LoadBalancingRouter`) rather than this engine's own
        least-utilized-available pop in `allocate_next`.

        Does exactly the same state mutation as `allocate_next` -
        removes the job from the waiting queue, updates GPU/User/
        Assignment state, logs the ALLOC/STATUS events - the only
        difference is *who* chose ``gpu``. This is the seam that keeps
        "which job" (this engine) and "which GPU" (the router)
        decisions separate without duplicating the commit logic in
        both places. Pass ``now`` (e.g. a simulated clock's current
        time) to keep every timestamp this call produces consistent
        with the caller's notion of "now" rather than the real wall
        clock.
        """
        return self._assign_gpu(job, gpu, policy, reason, candidates, now=now)

    def allocate_next(self, now: Optional[datetime] = None) -> Optional[AllocationDecision]:
        """Take one available GPU and give it to the winning waiting job.

        Returns ``None`` (and changes nothing) if there is no
        available GPU, or no waiting job to give it to - in the
        latter case the GPU is put back as available rather than lost.
        """
        gpu = self._pop_available_gpu()
        if gpu is None:
            return None

        selection = self.select_next_job()
        if selection is None:
            self._available_gpus.insert_gpu(gpu)
            return None

        winner, policy, reason, candidates = selection
        return self._assign_gpu(winner, gpu, policy, reason, candidates, now=now)

    def allocate_all(self, now: Optional[datetime] = None) -> List[AllocationDecision]:
        """Repeatedly allocate until no GPU is available or no job is
        waiting - Scenario 7's "drain the queue" behavior."""
        decisions: List[AllocationDecision] = []
        while True:
            decision = self.allocate_next(now=now)
            if decision is None:
                break
            decisions.append(decision)
        return decisions

    def release_user_gpu(self, user_id: str, gpu_id: str) -> None:
        """Remove a stale user -> GPU mapping from this engine's
        HashMap-backed index.

        `finalize_assignment`/`_assign_gpu` keep `_user_index` in sync
        for every assignment *this engine* creates, but an assignment
        can also end somewhere this engine never sees - a legitimate
        reclaim, decided entirely by `ReclamationEngine`. Without this
        call, `get_gpus_for_user` would keep reporting a GPU the user
        no longer holds (an audit found exactly this: the HashMap
        still listed a GPU that `GPU.assigned_user_id`/
        `User.assigned_gpu_ids` correctly showed as freed). Whoever
        ends an assignment outside this engine is responsible for
        calling this - `ReclamationEngine._reclaim` does.
        """
        self._user_index.remove_assignment(
            GPUAssignment(assignment_id="release", gpu_id=gpu_id, user_id=user_id)
        )

    # -- Read-only helpers -----------------------------------------------

    def get_gpus_for_user(self, user_id: str) -> List[str]:
        """O(1) hashmap lookup of the GPU ids currently held by a user."""
        return self._user_index.get_gpus_for_user(user_id)

    def waiting_job_ids(self) -> List[str]:
        """Currently-waiting job ids, in arrival order."""
        return [job.job_id for job in self._current_waiting_jobs()]

    def available_gpu_count(self) -> int:
        """How many GPUs are genuinely free right now.

        100-scenario validation, Phase 13: this used to report
        `self._available_gpus.size()` (the min-heap this engine's own
        `allocate_next`/`_pop_available_gpu` maintains) - but
        `Scheduler.try_allocate_all`'s real path (`LoadBalancingRouter`
        -> `finalize_assignment`) never pops from that heap at all, so
        the heap's size only ever grows and silently drifts from the
        truth the moment any GPU is actually committed through the
        Scheduler. `SchedulerState` is this project's one authoritative
        source of truth (every other phase already treats it that way);
        this now reads directly from it instead of trusting a second,
        independently-maintained count that nothing keeps in sync.
        """
        return sum(1 for gpu in self.state.gpus.values() if is_gpu_available(gpu))

    @property
    def gpu_pool(self) -> GPUPool:
        """The full GPU inventory (the Phase 2 linked list), for callers
        - like Phase 5's `LoadBalancingRouter` - that need to look across
        every known GPU rather than only this engine's own "already
        confirmed available" heap.
        """
        return self._pool
