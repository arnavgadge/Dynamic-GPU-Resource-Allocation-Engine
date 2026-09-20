# DSA Architecture

> A structure-by-structure reference for the data structures and
> algorithms used in the Dynamic GPU Resource Allocation Engine. For
> the full phase-by-phase narrative (including how these structures
> plug into the Allocation/Reclamation/Balancing engines), see
> [`architecture.md`](architecture.md). All implementations live under
> [`engine/dsa/`](../engine/dsa/), each with dedicated unit tests under
> [`tests/dsa/`](../tests/dsa/).

Every structure below exists because a specific piece of the
scheduler needs it — not to check a syllabus box. Two are used exactly
as the generic textbook structure (`HashMap`, `LinkedList` via
`GPUPool`); the others are generic building blocks with a thin,
project-specific wrapper on top so the underlying DSA component stays
reusable while the wrapper carries the project's vocabulary.

## The Dynamic GPU Pool

```
Company GPU Pool
       ↓
Dynamic GPU Pool        (engine.dsa.gpu_pool.GPUPool)
       ↓
Linked List             (engine.dsa.linked_list.LinkedList[GPU])
       ↓
GPU Nodes                (one node per GPU object)
       ↓
Scheduler / Allocation Engine
```

**Why a Linked List?**

- **Dynamic insertion/deletion of GPU nodes** — GPUs join and leave
  the company pool at arbitrary times (a new card provisioned, one
  decommissioned, an admin's maintenance action); a linked list's
  O(1) append/prepend and node-level removal fit that directly.
- **No need for contiguous storage** — unlike a fixed-size array, the
  pool never needs to be pre-sized or reallocated/shifted as GPUs are
  added or removed.
- **Suitable for a dynamically changing resource pool** — the pool's
  size is never assumed anywhere; `GPUPool` only ever reports
  `size()`/`is_empty()` from what is actually in the list right now.

**Complexity** (from `LinkedList`, unchanged by the `GPUPool` wrapper):

| Operation | Complexity | Why |
|---|---|---|
| `add_gpu` (append) | O(1) | Tail pointer — no traversal needed. |
| `remove_gpu` (by id) | O(n) | Must scan to find the matching node — the documented cost of a singly linked list without a reverse index. |
| `get_gpu` (find, by id) | O(n) | Same reason as removal. |
| `all_gpus` / iteration | O(n) total | One pass over every node. |
| `size` / `is_empty` | O(1) | Maintained by a counter, never counted on demand. |

**Determining available vs. allocated GPUs** is deliberately *not* a
second definition invented inside the pool — it composes the pool's
traversal with the project's one existing availability predicate,
`engine.balancing.availability.is_gpu_available` (`not gpu.is_assigned
and gpu.status == GPUStatus.IDLE`):

```python
available = [g for g in pool if is_gpu_available(g)]   # O(n)
allocated = [g for g in pool if g.is_assigned]          # O(n)
```

Keeping this logic in `balancing/`, not in `dsa/gpu_pool.py`, matters:
`dsa/` is the project's lowest layer (it depends only on `engine.models`),
and `balancing/` depends on `dsa/` — importing `balancing` back into
`dsa` would invert that dependency and risk a circular import. The
pool stays a pure data-structure wrapper; availability stays a policy
decision made in exactly one place, as documented in
`engine/balancing/availability.py`.

**GPU statuses and the pool.** The pool holds `GPU` objects exactly as
`engine.models.enums.GPUStatus` defines them — it introduces no
competing status vocabulary. Mapped onto this task's plain-English
terms:

| Plain-English term | Actual `GPUStatus` |
|---|---|
| AVAILABLE | `IDLE` |
| ALLOCATED | `ACTIVE` |
| IDLE_WARNING | `IDLE_WARNING` (unchanged) |
| RECLAIMING | `RECLAIMING` (unchanged) |
| MAINTENANCE | `MAINTENANCE` (unchanged) |
| UNAVAILABLE | `UNAVAILABLE` (unchanged) |

No new statuses were needed or added — the existing enum already
covers every state this task calls for, including the two that must
never be allocated (`MAINTENANCE`, an admin's deliberate choice via
`Scheduler.set_gpu_maintenance`; `UNAVAILABLE`, the hardware layer
reporting a GPU as gone via `Scheduler.handle_gpu_failure`). Both are
excluded from `is_gpu_available` simply by not being `IDLE` — no
special-case branching was required.

**GPU lifecycle**, exactly as the existing engines already drive it:

```
AVAILABLE (IDLE)
   ↓  AllocationEngine commits a job
ALLOCATED (ACTIVE)
   ↓  Scheduler.complete_job / a reclaim resolves
RELEASED (GPU.assigned_user_id/job_id -> None, status -> IDLE)
   ↓
AVAILABLE (IDLE) again
```

**Arbitrary pool size.** Nothing in `GPUPool`, `AllocationEngine`, or
`Scheduler` reads or branches on a GPU count. The current
demonstration initializes 5 logical GPUs (`GPU-1`..`GPU-5`); the same
code path is exercised at 1, 10, 50, and 100 GPUs in
`tests/dsa/test_gpu_pool.py` with no engine changes of any kind — the
number of GPUs is purely a property of how many `GPU` objects a
caller happens to `add_gpu()`.

| Data Structure | File(s) | Purpose in this project | Complexity |
|---|---|---|---|
| **Linked List** | `dsa/linked_list.py` (generic) → `dsa/gpu_pool.py` (`GPUPool`) | The company's GPU inventory. A linked list gives O(1) insertion/removal as GPUs are dynamically added to or removed from the pool — no shifting elements the way a fixed array would need. | append/prepend O(1); find/remove O(n); size O(1) |
| **Min-Heap** | `dsa/min_heap.py` (generic) → `dsa/gpu_utilization_heap.py` (`GPUUtilizationHeap`) | Efficiently answers "which GPU is least utilized right now?" — used by the Load-Balancing Router to pick the least-utilized *available* GPU. Never decides whether a GPU should be reclaimed. | insert O(log n); extract-min O(log n); peek O(1); rebuild (after external mutation) O(n) |
| **Max-Heap** | `dsa/max_heap.py` (generic) | The ordering engine `PriorityQueue` is built on. Kept separate and comparator-driven so no priority *rule* is baked into the heap itself. | insert O(log n); extract-max O(log n); peek O(1); rebuild O(n) |
| **Priority Queue** | `dsa/priority_queue.py` (`PriorityQueue`, wraps `MaxHeap`) | Manages waiting jobs by whatever comparator the scheduler supplies (the allocation-score formula) — insertion-order tie-breaking for equal priorities. | insert O(log n); pop-best O(log n); peek O(1) |
| **HashMap** | `dsa/hashmap.py` (generic) → `dsa/user_gpu_index.py` (`UserGPUIndex`) | Fast user → currently-assigned-GPU(s) lookup, built from `GPUAssignment` records; a real bucket array with separate-chaining collision handling, doubling past a 0.75 load factor. | put/get/remove/contains O(1) average (amortized resize); O(n) worst case |
| **Queue** | `dsa/queue.py` (generic) → `dsa/waiting_job_queue.py` (`WaitingJobQueue`) | Preserves arrival order for waiting jobs, backing the FCFS path when the 20%-similarity rule decides two jobs should be served by arrival order instead of by score. | enqueue/dequeue/peek O(1) |
| **Stack** | `dsa/stack.py` (generic) → `dsa/reclaim_history.py` (`ReclaimHistory`) | Records reclaim `Event`s so the most recent reclaim is always what a rollback would undo first. | push/pop/peek O(1) |
| **Sliding Window** | `dsa/sliding_window.py` (`UtilizationSlidingWindow`, built on `Queue`) | Keeps only the `UtilizationObservation`s inside a trailing time window, so the reclamation engine can tell a *sustained* low reading apart from a brief dip, without rescanning a GPU's entire history each time. | add O(1) + amortized O(1) eviction per observation; window query O(n) in current window; span O(1) |

## GPU Utilization Tracking (the Min-Heap)

```
NVML / nvidia-smi
        ↓
Hardware Monitor          (engine.hardware.monitor.GPUMonitor - one interface)
        ↓
MonitorPoller              (engine.hardware.poller - feeds readings into the scheduler)
        ↓
SchedulerState GPU utilization   (GPU.utilization_percent - the one source of truth)
        ↓
GPU Utilization Min-Heap    (engine.dsa.gpu_utilization_heap.GPUUtilizationHeap / a router-built MinHeap)
        ↓
Scheduler / Allocation Engine
```

For simulation, the exact same pipeline runs on a mock source instead
of real hardware — nothing below `MonitorPoller` knows or cares which
one fed it:

```
Mock Monitor (SimulatorGPUMonitor)
        ↓
MonitorPoller
        ↓
same SchedulerState
        ↓
same Min-Heap
        ↓
same Scheduler
```

**Why a Min-Heap here.** The engine frequently needs to answer "which
suitable GPU currently has the lowest utilization?" — for routing new
work, that question would otherwise mean scanning every GPU in the
pool on every decision. A Min-Heap answers it in O(log n) per
insertion and O(1) to peek, instead of an O(n) scan repeated on every
allocation.

**Where the Min-Heap is actually used** — two real sites, not one
generic index reused blindly:

1. **`LoadBalancingRouter.route_job`** (`engine/balancing/router.py`)
   — the live path every real allocation actually goes through
   (`Scheduler.try_allocate_all`). It builds a fresh `MinHeap[GPU]`,
   keyed on `(utilization_percent, gpu_id)`, from whatever is
   currently available *at the moment of the decision* — never a
   heap that could hold a stale reading, because it is rebuilt from
   `SchedulerState` on every single call. The `gpu_id` tie-break makes
   ties deterministic rather than dependent on dict/insertion order.
2. **`AllocationEngine`'s `GPUUtilizationHeap`** (`_available_gpus`) —
   a *persistent* heap backing the engine's own standalone
   `allocate_next`/`allocate_all` (used directly by `main.py`'s demo
   and by allocation-only unit tests, not by `Scheduler`'s own live
   path). Because it is persistent, a GPU's utilization can change
   while it is sitting in the heap; `GPUUtilizationHeap.refresh()`
   (built on `MinHeap.rebuild`, O(n)) is the project's chosen update
   mechanism for that — verified directly in
   `tests/dsa/test_gpu_utilization_heap.py`.

**Utilization updates and consistency.** `GPU.utilization_percent`
lives on the one `GPU` object `SchedulerState` owns; every heap above
holds *references* to that same object, never a copy. This project
deliberately does not maintain a second, independent utilization
value inside either heap — an update always means either (a) rebuild
the heap fresh from `SchedulerState` (the router's approach, and the
one used for every live allocation), or (b) call `refresh()` on a
persistent heap after a referenced GPU's key changed (the
`AllocationEngine`/`main.py`-demo approach). Both keep `SchedulerState`
as the single authoritative source; the heap is always just an
efficient index over it, never a competing truth.

**Available-GPU-count consistency.** `AllocationEngine.
available_gpu_count()` deliberately does **not** read
`self._available_gpus.size()` — that heap only ever grows relative to
GPUs actually committed through `Scheduler.try_allocate_all` (which
never pops from it), so trusting it would silently drift from the
truth the moment any GPU is committed through the live path (this was
a real issue an earlier validation pass found and fixed). It instead
counts directly over `SchedulerState.gpus` using the one
`is_gpu_available` predicate — the same guarantee
`test_available_gpu_count_matches_state_even_when_the_legacy_heap_has_drifted`
locks down by deliberately manufacturing that drift and asserting the
count still matches ground truth.

**Complexity**, unchanged from `MinHeap`'s own documented cost:

| Operation | Complexity |
|---|---|
| `peek_least_utilized` / `peek_min` | O(1) |
| `insert_gpu` / `insert` | O(log n) |
| `extract_least_utilized` / `extract_min` | O(log n) |
| `refresh` / `rebuild` (after an already-inserted GPU's utilization changed) | O(n) |

## User/Job Management: HashMap Lookup & Waiting Queue

**Which structure is authoritative, and which are indexes.**
`SchedulerState` (`engine/models/scheduler_state.py`) is the one
source of truth: `gpus`/`users`/`jobs` are plain Python `dict`s keyed
by id. A `dict` *is* a hash map — this project deliberately does not
reimplement a hand-built `HashMap` here a second time (see "Why the
HashMap is hand-built instead of `dict`" below for where a hand-built
one *is* worth it, and why); `SchedulerState.get_user`/`get_job`/
`get_gpu` are already the O(1)-average `user_id -> User` /
`job_id -> Job` / `gpu_id -> GPU` lookups this task asks for, and they
are what every engine (`AllocationEngine`, `ReclamationEngine`,
`LoadBalancingRouter`, `Scheduler`) already reads from — not a second,
competing lookup path built for this task.

The one place a genuinely different-shaped index earns a hand-built
`HashMap` is **`UserGPUIndex`** (`dsa/user_gpu_index.py`) — a
`user_id -> List[gpu_id]` *reverse* index (one user can hold many
GPUs), built from `GPUAssignment` records and kept live by
`AllocationEngine._user_index`. It already participates in the real
scheduler path (`AllocationEngine.get_gpus_for_user`, `release_user_gpu`)
rather than existing only for demonstration.

**GPU -> Job -> User (the reverse chain).** `SchedulerState.
get_job_on_gpu(gpu_id)` (one dict lookup by `gpu.assigned_job_id`,
already O(1)) chained with `get_user(job.user_id)` (another O(1) dict
lookup) answers "GPU-2 -> Job 101 -> User A" in two O(1) hops, with no
new structure needed.

**User -> Job(s).** Two existing, already-used equivalents, for two
different questions: `User.running_job_ids` (O(1) list read) for
*currently running* jobs, and filtering `SchedulerState.jobs` by
`job.user_id` (O(n) over all jobs) for *every* job regardless of
status — the exact pattern `api/serializers.py::serialize_portal_state`
already uses to build a user's "my jobs" view. Job/user counts are
small at this project's scale (the same reasoning already documented
for several O(n) operations elsewhere in this file), so this was not
worth a dedicated reverse index.

**The waiting queue.** `WaitingJobQueue` (`dsa/waiting_job_queue.py`,
wrapping the generic `Queue`) is the actual FIFO storage
`AllocationEngine` enqueues every waiting job into and removes them
from — not a parallel structure kept in sync by hand. A job that
cannot yet be fully satisfied (a multi-GPU request short of
`gpu_count`) stays as **one** queue entry, never one per missing GPU —
`Job.gpu_count` (requested), `len(Job.assigned_gpu_ids)` (allocated),
and `Job.gpus_still_needed` (remaining) already represent that state
cleanly on the job itself; the queue only ever holds *jobs*, never a
per-GPU placeholder.

**Job lifecycle**, using the existing `JobStatus` enum only — no new
states were added:

```
(REQUEST event logged) -> WAITING -> RUNNING -> COMPLETED
                              |          |
                              v          v
                          CANCELLED   RECLAIMED
```

"REQUESTED" (from the task's own wording) is not a persisted
`JobStatus` — it is the `EventType.REQUEST` event `Scheduler.submit_job`
already logs at the moment of submission, immediately followed by the
job's initial `JobStatus.WAITING`. Priority preemption and a GPU
holding only *some* of a multi-GPU job's GPUs failing both still end
at `RECLAIMED`/`WAITING` respectively through the project's one real
reclaim path (`ReclamationEngine._reclaim`/`Scheduler.handle_gpu_failure`)
— distinguished by the *event*'s type/metadata
(`PRIORITY_PREEMPTION`/`HARDWARE_FAILURE`), never by inventing a
`PREEMPTED` or `FAILED` job status the enum's own documented design
already rejected ("a separate interrupted vs. reclaimed state was
considered and dropped").

**A real consistency bug found and fixed while verifying this.**
`Scheduler.complete_job` released a finished job's GPU from `GPU`/
`User` correctly, but — unlike `force_reclaim`, `handle_gpu_failure`,
and `ReclamationEngine._reclaim`, which all call
`AllocationEngine.release_user_gpu` — it never told the HashMap-backed
`UserGPUIndex`. `AllocationEngine.get_gpus_for_user` would then keep
reporting a GPU a user no longer held, after an ordinary job
completion specifically. Fixed by adding the same `release_user_gpu`
call every other release path already makes — one line, no policy
change; regression-tested in
`tests/test_user_job_management.py::test_no_dangling_userindex_entry_after_normal_completion`.

**Complexity:**

| Structure | Operation | Complexity |
|---|---|---|
| `SchedulerState` dict lookups | get by id | O(1) average |
| `HashMap` / `UserGPUIndex` | get/put/remove | O(1) average (amortized resize); O(n) worst case |
| `Queue` / `WaitingJobQueue` | enqueue/dequeue/peek | O(1) |
| User -> all jobs (status-agnostic) | filter `SchedulerState.jobs` | O(n) in total job count |

## Allocation Engine & Weighted Scoring

**Flow** (all backend; `Scheduler.try_allocate_all` is the one place it
is sequenced, and it makes no decision itself):

```
Waiting Jobs (WaitingJobQueue, FIFO)
     ↓
Critical-tier gate  →  20% size-similarity check  →  FCFS  or  score-based
     ↓ (score-based)
Calculate Scores        engine/allocation/scoring.py
     ↓
Priority Queue / Max-Heap    (PriorityQueue over MaxHeap)
     ↓
Highest-scoring job (ties: earliest submitted_at)
     ↓
GPU Pool + Min-Heap     LoadBalancingRouter: available GPUs only, least utilized first
     ↓
finalize_assignment     GPU/User/Job/GPUAssignment/UserGPUIndex updated together
     ↓
ALLOC + STATUS events   (and the decision trace)
```

**Formula.** `Allocation Score = 0.6 × priority component + 0.4 × size
component + aging`. Each term is its own function in `scoring.py` —
`calculate_priority_component` (fixed 0..1 scale from the `Priority`
enum), `calculate_size_component` (min-max inverse *among the current
candidates*, never `1/size`), `calculate_aging_component` (0.01 per
minute waited, capped at 1.1) — combined only by
`calculate_allocation_score`, which returns a `ScoreBreakdown`. The
weights and aging constants live once, in `allocation/config.py`.
`CandidateInfo` (the per-candidate record on every `AllocationDecision`)
now carries the full explanation: priority, size, waiting time,
priority component, size component, aging contribution, base and
final score.

**Priority Queue integration.** The score-based branch inserts the
candidates into a `PriorityQueue` keyed on `(final_score,
-submitted_at)` and pops the winner — higher score first, equal scores
resolved deterministically by earliest submission. The candidate set
is rebuilt from the live waiting queue on every decision, so the
queue can never disagree with `SchedulerState` when a job enters,
leaves, is cancelled, is allocated, is requeued, or changes priority
(`Job.priority` is read at decision time; there is no stale
priority snapshot to re-insert). Selection also filters on
`JobStatus.WAITING`, so a cancelled job left in the FIFO is never
offered.

**Multi-GPU.** A job is one queue entry whose `gpu_count` /
`len(assigned_gpu_ids)` / `gpus_still_needed` track the request. Each
pass commits one GPU at a time under the same policy; the job becomes
`RUNNING` only when fully allocated (requested 4 → 2/2 → 3/1 → 4/0).
`requeue_job` is idempotent — a partially-allocated job that loses its
last GPU is not queued twice (a duplicate-entry bug found and fixed
on Day 5).

**Complexity** (n = candidates in the decision, g = available GPUs):

| Step | Structure | Cost |
|---|---|---|
| Insert scored candidates | Max-Heap / Priority Queue | O(log n) each |
| Highest-scoring job | Max-Heap peek / pop | O(1) peek, O(log n) pop |
| Remove a waiting job | FIFO rebuild (`_remove_from_waiting_queue`) | O(w), w = jobs waiting |
| Least-utilized available GPU | Min-Heap (router) | O(log g) per insert/extract, O(1) peek |
| id → User / Job / GPU | `SchedulerState` dicts (hash maps) | O(1) average |
| Whole decision | select + route + commit | O(n log n + g log g) |

The three structures cooperate in one path: the Priority Queue picks
*which job*, the Min-Heap picks *which GPU*, and the hash maps resolve
the ids both decisions touch without scanning.

## Adaptive FCFS / SJF Scheduling

**The rule.** Before any scoring, `AllocationEngine.select_next_job`
asks one question of the waiting jobs (after the CRITICAL-tier gate
has narrowed the pool): `are_job_sizes_similar(jobs)`
(`allocation/similarity.py`). The metric is the relative size spread
`(largest − smallest) / largest`; the threshold is
`JOB_SIZE_SIMILARITY_THRESHOLD` (0.20, `allocation/config.py`) and can
be overridden per call (`threshold=`). A spread exactly at 20% counts
as similar.

| Spread | Mode | Winner |
|---|---|---|
| ≤ 20% (10 min vs 12 min → 16.7%) | **FCFS** | earliest `submitted_at` (ties: queue order) |
| > 20% (15 min vs 60 min → 75%) | **SJF / Weighted** | highest blended score via the Priority Queue (ties: earliest `submitted_at`) |

**Blended, not tiered.** Neither mode is a hard `HIGH > MEDIUM > LOW`
gate. In weighted mode priority is 60% of the score, size 40%, plus
aging — so a much smaller MEDIUM job can beat a large HIGH one (the
`ml_video` scenario), and in FCFS mode arrival order decides
regardless of priority (`test_engine.py::test_scenario_2`). Only
`CRITICAL` narrows the candidate pool first. This is the project's
deliberate policy; the Day 6 tests pin it.

**Aging coexists.** Aging is one term inside the weighted score (0.01
per minute, capped at 1.1), so it acts exactly where scores are
computed: a long-waiting job can overtake a fresher, better-scoring
one in weighted mode. FCFS mode never needs it — the longest waiter
already wins — and reports `score = None` because no score was
computed.

**Decision trace.** Every `AllocationDecision` now carries the numbers
behind the mode choice — `size_spread`, `similarity_threshold`, each
candidate's `arrival_position`, priority, size, waiting time and (in
weighted mode) priority/size/aging components and final score — and
`decision.explain()` renders them:

```
Policy: SJF / Weighted
Size spread: 75.0% (beyond the 20% threshold)
Reason: no critical jobs waiting - scoring open to all waiting jobs; job sizes differ beyond 20% (spread=75.0%) -> score-based selection; A scored highest (0.690, incl. +0.090 aging) | routed: GPU-1 has the lowest utilization (0.0%) among 1 available GPU(s)
Candidates:
  #1 A user=U-A MEDIUM size=15min waited=0:09:00 score 0.690 = base 0.600 (priority component 0.333, size component 1.000) + aging 0.090
  #2 B user=U-B MEDIUM size=60min waited=0:08:00 score 0.280 = base 0.200 (priority component 0.333, size component 0.000) + aging 0.080
Selected: A -> GPU-1
```

**How each structure participates in the pipeline:**

| Structure | Role in the adaptive pipeline |
|---|---|
| `Queue` (`WaitingJobQueue`) | Holds the waiting jobs in arrival order; FCFS ties resolve by this order. |
| Priority Queue / Max-Heap | Picks the highest-scoring candidate in weighted mode. |
| HashMap / `SchedulerState` dicts | Resolve job → user → GPU ids during scoring and commit. |
| Min-Heap | After the job is chosen, picks the least-utilized *available* GPU. |
| Weighted scoring | Combines priority, size and aging (weighted mode only). |

Cost per decision: O(k) to compute the spread and pick the FCFS
winner, or O(k log k) to score and heap-order k candidates, then
O(g log g) for GPU routing.

## Two extra project concepts, not separate data structures

- **Weighted scoring** — the allocation-score formula
  (`engine/allocation/scoring.py`) combines a normalized priority
  component and a normalized job-size component (60/40 by default) to
  rank waiting jobs. Not a data structure itself, but the comparator
  the Priority Queue / score-based selection path is built around.
- **SJF (job-size-based scheduling)** — the "size" half of the
  allocation score, plus the 20% job-size-similarity check
  (`engine/allocation/similarity.py`) that decides whether a round of
  competing jobs is served FCFS or by score at all.
- **Load balancing** — `engine/balancing/router.py` is the algorithm
  that consumes the Min-Heap above (plus a HashMap lookup and the
  Linked-List-backed GPU pool) to route new work to the least-utilized
  *available* GPU, never touching a GPU that is already legitimately
  assigned regardless of how idle it looks.

## Why a Max-Heap *and* a Priority Queue, when they sound like the same thing?

They're not duplicated — they're layered. `MaxHeap[T]` is the raw,
comparator-free array-heap mechanism (sift-up/sift-down).
`PriorityQueue[T]` is that mechanism wrapped with a caller-supplied
`key_fn` and FIFO tie-breaking. Building the Priority Queue *on top
of* the Max-Heap, rather than writing two independent heaps, is what
keeps the Max-Heap itself free of any hardcoded priority rule.

## Why the GPU pool is a real linked list and not `list.append`

Python's `list` already gives amortized O(1) append, so the
difference isn't raw speed — it's that a `list` doesn't teach or
demonstrate anything about node-based structures, and the project
explicitly calls for a linked list here. `LinkedList` is implemented
with actual node objects and head/tail pointers; `GPUPool` is the
thin, GPU-flavoured API around it.

## Why the HashMap is hand-built instead of `dict`

Same reasoning: `dict` would work, but wrapping it teaches nothing.
`HashMap` is a real bucket array with separate-chaining collision
handling and doubles its capacity past a 0.75 load factor, same as
production hash tables do it.

## Why `Queue` uses head/tail node pointers but `Stack` uses a plain list

A queue must add at one end and remove from the other; `list.pop(0)`
for that would be O(n) because every remaining element shifts down, so
`Queue` needs real linked nodes for O(1) on both ends. A stack only
ever touches one end, which is exactly what `list.append`/`list.pop()`
(no index argument) already do in O(1) — adding node-linking there
would add complexity without changing the complexity class.

## Where each structure is actually wired in

| Structure | Consumed by |
|---|---|
| `GPUPool` (linked list) | `AllocationEngine` — the full GPU inventory (`add_gpu`/`remove_gpu`/lookup); `LoadBalancingRouter` draws its candidate pool from the same object, never a second one. |
| `GPUUtilizationHeap` (min-heap) | `AllocationEngine._pop_available_gpu` — holds only currently-available GPUs. `LoadBalancingRouter` builds an equivalent `MinHeap` over routable candidates for `route_job`. |
| `PriorityQueue` / `MaxHeap` | Available for score-ordered selection; `main.py`'s Phase 2 demo exercises it directly. `AllocationEngine.select_next_job` itself sorts the small, per-decision candidate list directly, since it's at most "however many jobs are waiting" — see `architecture.md`'s Allocation Engine section for why that isn't "bypassing the DSA layer". |
| `WaitingJobQueue` (Queue) | `AllocationEngine` — the actual storage for every waiting job in arrival order; the FCFS tie-break path reads from it. |
| `UserGPUIndex` (HashMap) | Updated on every assignment; `get_gpus_for_user` is an O(1) lookup of what a user currently holds. |
| `ReclaimHistory` (Stack) | `ReclamationEngine.last_reclaim()` / `undo_last_reclaim()` — most-recent-reclaim-first. |
| `UtilizationSlidingWindow` | One per watched GPU, sized to the larger reclamation tier's duration; `ReclamationEngine._detect_tier` reads it (never the GPU's full lifetime history) to run the sustained-breach check. |

Every structure is tested twice over: once directly (`tests/dsa/`),
proving the structure's own contract, and once indirectly, through the
engine/integration tests, proving it is actually wired into the
decision it claims to serve.
