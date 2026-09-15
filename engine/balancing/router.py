"""LoadBalancingRouter: decides WHERE a job should go, once something
else has already decided THAT it should go somewhere.

Phase 3's `AllocationEngine` answers "which waiting job should be
scheduled next?" - it owns the allocation-score formula, the 20%
similarity rule, critical-job precedence. This module answers a
completely different question: "given that job, which currently
*available* GPU should receive it, so new work naturally drifts
toward whichever GPU is least busy?" It never re-runs Phase 3's
scoring, never decides which job goes next, and - most importantly -
never touches a GPU that is already legitimately assigned, no matter
how low its utilization reads. Reclaiming a GPU because it looks idle
is Phase 4's job, not this one's; this module only ever routes *new*
work, and only onto GPUs that are already, genuinely available.

    Phase 3 (AllocationEngine)   ->  WHICH JOB?
    Phase 5 (LoadBalancingRouter) -> WHICH GPU (for a job Phase 3
                                       already chose to schedule)?
"""

import itertools
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from engine.balancing.availability import has_valid_utilization, is_gpu_available
from engine.balancing.config import BalancingPolicy, DEFAULT_BALANCING_POLICY
from engine.balancing.decision import GPUCandidateInfo, RoutingDecision, RoutingOutcome
from engine.balancing.detector import is_pool_imbalanced, utilization_spread
from engine.dsa.hashmap import HashMap
from engine.dsa.min_heap import MinHeap
from engine.models.enums import EventType
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState


class LoadBalancingRouter:
    """Chooses the best currently-available GPU for one job at a time.

    Reads `SchedulerState` but never mutates a GPU/Job/User/Assignment
    itself - `route_job` only *decides*. Committing the decision (the
    part that actually moves the job to RUNNING and the GPU to ACTIVE)
    is `AllocationEngine.finalize_assignment`, the same commit path
    Phase 3 already uses for its own self-contained allocation - so
    routed and non-routed assignments are indistinguishable in the
    resulting state, only in how the GPU was chosen.
    """

    def __init__(self, state: SchedulerState, policy: BalancingPolicy = DEFAULT_BALANCING_POLICY) -> None:
        self.state = state
        self.policy = policy
        self._event_seq = itertools.count(1)

    def route_job(
        self, job: Job, candidate_gpus: Optional[Iterable[GPU]] = None, now: Optional[datetime] = None
    ) -> RoutingDecision:
        """Pick a destination GPU for ``job`` among ``candidate_gpus``
        (every known GPU by default, from `SchedulerState.gpus`).

        Never invents a destination: if nothing is genuinely
        available, returns a decision with ``outcome ==
        RoutingOutcome.NO_GPU_AVAILABLE`` and ``selected_gpu_id ==
        None`` - the caller is expected to leave the job WAITING.
        """
        now = now or datetime.now(timezone.utc)
        pool = list(candidate_gpus) if candidate_gpus is not None else list(self.state.gpus.values())

        candidate_info = [
            GPUCandidateInfo(gpu_id=gpu.gpu_id, utilization_percent=gpu.utilization_percent,
                              available=is_gpu_available(gpu))
            for gpu in pool
        ]
        available_gpus = [gpu for gpu in pool if is_gpu_available(gpu)]
        available_info = [info for info in candidate_info if info.available]

        self._log_event(
            job, now, EventType.BALANCE, gpu_id=None,
            message=f"Evaluated {len(pool)} GPU(s) in the pool for {job.job_id} - {len(available_gpus)} available",
            reason="inspecting the available GPU pool for new-task routing",
        )

        # O(1) lookup back from an id to its GPU object, once a
        # decision is made - the same pattern `UserGPUIndex` (Phase 2's
        # HashMap wrapper) uses elsewhere in this project for O(1)
        # assignment lookups.
        gpu_by_id: HashMap[str, GPU] = HashMap()
        for gpu in available_gpus:
            gpu_by_id.put(gpu.gpu_id, gpu)

        routable_gpus = [gpu for gpu in available_gpus if has_valid_utilization(gpu)]
        unsafe_count = len(available_gpus) - len(routable_gpus)

        if not routable_gpus:
            reason = "no available GPU with a valid utilization reading - job remains WAITING"
            if unsafe_count:
                reason += f" ({unsafe_count} available GPU(s) had an invalid/unsafe utilization reading)"
            event = self._log_event(job, now, EventType.BALANCE, gpu_id=None,
                                     message=f"No routable GPU for {job.job_id}", reason=reason)
            return RoutingDecision(
                timestamp=now, job_id=job.job_id, candidates=candidate_info,
                available_candidates=available_info, selected_gpu_id=None,
                imbalance_detected=False, outcome=RoutingOutcome.NO_GPU_AVAILABLE,
                reason=reason, event=event,
            )

        imbalanced = is_pool_imbalanced(routable_gpus, self.policy)

        # Reuse the Phase 2 Min-Heap: availability (and utilization
        # safety) is filtered first in one O(n) pass over the pool;
        # the heap then finds the minimum in O(log n) per insertion
        # rather than a second full scan or a `sorted()` call. The key
        # is `(utilization_percent, gpu_id)` - a tuple - so two GPUs
        # tied on utilization are broken deterministically by GPU id
        # (lexicographic), never by dict/insertion-order happenstance.
        heap: MinHeap[GPU] = MinHeap(key_fn=lambda gpu: (gpu.utilization_percent, gpu.gpu_id))
        for gpu in routable_gpus:
            heap.insert(gpu)
        best = heap.extract_min()
        selected = gpu_by_id.get(best.gpu_id)

        spread = utilization_spread(routable_gpus)
        reason = (
            f"{selected.gpu_id} has the lowest utilization "
            f"({selected.utilization_percent:.1f}%) among {len(routable_gpus)} available GPU(s)"
        )
        if imbalanced:
            reason += (
                f"; available-pool spread {spread:.1f}pp >= "
                f"{self.policy.imbalance_threshold_percent:.0f}pp threshold - meaningful imbalance"
            )

        event = self._log_event(
            job, now, EventType.BALANCE, gpu_id=selected.gpu_id,
            message=f"{selected.gpu_id} selected as lowest-utilization available GPU for {job.job_id}",
            reason=reason,
        )

        return RoutingDecision(
            timestamp=now, job_id=job.job_id, candidates=candidate_info,
            available_candidates=available_info, selected_gpu_id=selected.gpu_id,
            imbalance_detected=imbalanced, outcome=RoutingOutcome.ROUTED,
            reason=reason, event=event,
        )

    def _log_event(
        self, job: Job, now: datetime, event_type: EventType, gpu_id: Optional[str], message: str, reason: str
    ) -> Event:
        event = Event(
            event_id=f"B{next(self._event_seq)}",
            event_type=event_type,
            timestamp=now,
            gpu_id=gpu_id,
            user_id=job.user_id,
            job_id=job.job_id,
            message=message,
            reason=reason,
        )
        self.state.log_event(event)
        return event
