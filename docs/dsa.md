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

## GPU Reclamation & the Sliding Window

**Why a Sliding Window.** A single utilization reading is noise: a GPU
at 2% for one sample may just be loading its next batch. Reclamation
must judge a *span of time*, so `UtilizationSlidingWindow` (built on the
FIFO `Queue`) keeps only the readings inside a trailing window sized to
the longest tier (2h30), evicting from the front as new samples arrive
— amortized O(1) per reading, and the check never rescans a GPU's whole
history. `is_sustained_breach` then reasons over **timestamps**, not
counts, so irregular sampling is handled correctly: five readings in two
minutes are not "sustained", and a monitoring gap does not count as
evidence.

| Tier | Condition (continuous, up to the latest reading) | Default |
|---|---|---|
| 1 | utilization **< 2%** for the full duration | 25 min |
| 2 | utilization **< 15%** for the full duration | 2 h 30 min |

Exactly-at-threshold values (2.0%, 15.0%) are not "below"; exactly the
full duration counts as sustained. Tier 1 is checked first. Any reading
at/above the threshold inside the window breaks the streak — brief dips
(`80, 78, 2, 3, 75, 82`) never trigger anything.

**A GPU's history belongs to its holder.** Breach detection only counts
readings taken since the *current* holder's tenure began
(`_GPUWatch.holder_since`, set when the engine first sees a job on the
GPU and cleared when it is unassigned), after the last YES/reclaim
(`confirmed_at`), and `clear_watch_for_gpu` restarts the baseline on
completion/force-reclaim/failure even with no prompt pending. Otherwise
idle readings from an unassigned GPU or a completed job would make a
brand-new holder look "sustained idle" within minutes (a bug found and
fixed on Day 7).

**State machine** (one machine, reused by every trigger — tiers,
estimated completion, resource requests, priority preemption):

```
ACTIVE ─(sustained breach)→ IDLE_WARNING + PROMPT event
   ├─ YES ───────→ ACTIVE   (timer reset; RESPONSE + STATUS events)
   ├─ NO ────────→ RECLAIM ─→ IDLE
   └─ no reply for 5 min ──→ RECLAIM ─→ IDLE   (`no_response_grace_period`)
IDLE ─(Scheduler.try_allocate_all → AllocationEngine)→ ACTIVE  (ALLOC + STATUS events)
```

`IDLE_WARNING` is a `GPUStatus`, announced by the PROMPT event;
"reallocation" is the ordinary ALLOC event that follows the RECLAIM —
no separate event types were invented. The prompt is addressed to the
affected user (`event.user_id`), and the grace period is 5 **minutes**
(pinned by a test, including the live `SimulationSession`).

**Reclaim history.** Every reclaim pushes its RECLAIM event onto
`ReclaimHistory` (a `Stack`, most recent first). The event carries GPU,
user, job, reason, timestamp, and `metadata`: `category`
(`AUTOMATIC_RECLAIM`, `RESOURCE_REQUEST` or `PRIORITY_PREEMPTION`),
`previous_status` and `resulting_status`. Normal completion
(`JOB_COMPLETION`), admin release (`ADMIN_FORCE_RECLAIM`) and
`HARDWARE_FAILURE` are distinct events and never enter the history.

**Partial multi-GPU.** Reclamation works per GPU: only a GPU that itself
stays idle long enough is prompted, so a 4-GPU job with two busy GPUs
keeps those two (and stays RUNNING) while the two idle ones return to
the pool and serve a waiting job, which may still need more.

**Structures in the path:** Sliding Window → sustained detection; Stack
→ reclaim history; hash maps → GPU/job/user resolution during the
reclaim; Priority Queue + Queue → who receives the freed GPU; Min-Heap
→ which free GPU the router hands them.

**Not done (noted):** `Scheduler.respond_to_prompt` and
`check_reclamation_timeouts` reclaim but do not themselves call
`try_allocate_all`; the API session, simulator and `MonitorPoller` do
that immediately afterwards. There is also no `MONITOR` event on the
live path (only in sample data) — logging one per reading would be
noise.

## GPU Load Balancing & Hysteresis

**Two separate mechanisms, kept apart on purpose:**

```
GPU utilization
      ↓
Identify suitable resources     is_gpu_available (IDLE + unassigned only)
      ↓
Compare current utilization     Min-Heap, keyed (utilization_percent, gpu_id)
      ↓
Select appropriate GPU          LoadBalancingRouter.route_job
      ↓
Allocate NEW work                AllocationEngine.finalize_assignment
      ↓
Update utilization/state         GPU/User/Job/UserGPUIndex + ALLOC/STATUS events
```

1. **Routing new work** (`LoadBalancingRouter.route_job`, unchanged
   since Phase 5) - a job that needs a GPU is given the least-utilized
   *available* one via a fresh Min-Heap built each call.
   `is_gpu_available` excludes anything not plainly `IDLE` -
   `MAINTENANCE`, `UNAVAILABLE`, and any GPU still `is_assigned` -
   so an actively-used GPU (even at 4% utilization) is never a
   candidate, and this path never needs a cooldown of its own: it
   only ever touches genuinely free GPUs, and a GPU it just routed to
   is immediately assigned, so it cannot be reconsidered again while
   busy.

2. **Reallocation asks**
   (`Scheduler._request_additional_gpus_if_needed`, via
   `ReclamationEngine.request_gpu_for_reallocation`) - asking
   *another user* to release a GPU: an underutilized one for a
   multi-GPU deficit, or a lower-priority one under priority
   preemption. This never invents a second reclamation mechanism - it
   raises the exact same confirmation prompt every other trigger uses,
   and only a real YES/NO (or timeout) actually moves the GPU.

**Min-Heap is a candidate filter, not the whole policy.** The router's
heap only ever contains GPUs that already passed availability and a
valid-utilization check - "lowest utilization" is the tie-break among
*already-eligible* candidates, never a substitute for eligibility
itself.

**Hysteresis / cooldown** protects path 2 only, because that is the
only path that can ask about the *same* GPU more than once.
`BalancingPolicy.preemption_cooldown` (default 10 minutes, configurable) is
enforced by `ReclamationEngine.is_in_cooldown`, timed from
`cooldown_start` = the moment a prompt on that GPU was last *resolved*
(YES or NO) to `cooldown_expiry` = `cooldown_start + preemption_cooldown`;
a fresh ask on that GPU is blocked until `now >= cooldown_expiry`.
It is entirely `now`-based (the project's simulated clock in every
test), never a call counter, so any number of balancing-triggering
calls inside the window are all blocked alike, and utilization
oscillating near a threshold (4/8/5/7/4/9%) cannot cause repeated
asks. Cooldown is **per GPU**, blocks *any* job's ask on that GPU (not
only the original decliner's), and is distinct from
`has_declined_for` (which remembers a specific job was told no,
forever, until that job stops needing GPUs).

**Cooldown never blocks ordinary allocation.** `route_job` never
consults cooldown at all - once a GPU is genuinely `IDLE` (released
via NO, reclaimed, or completed), it is immediately routable to any
new job, even while `is_in_cooldown` still reports true for
reallocation-ask purposes. Cooldown only ever prevents a *repeated
ask*; it never marks a GPU unavailable, and never outlives the GPU's
own real state.

**Multi-GPU load balancing** reuses the existing allocation policy
per GPU: `try_allocate_all`'s loop calls the router once per
available GPU, so a 3-GPU request is satisfied by three independent,
eligibility-checked routing decisions, never a blind "three lowest
numbers" pick - a busy GPU at 1% is never chosen over free GPUs at
40/45/50%, because it never becomes a candidate in the first place.

**User isolation.** `GPU.utilization_percent` is a raw signal only;
whether a GPU is *available* depends solely on `is_gpu_available`
(assignment + status), never utilization. A GPU a user holds at 3% is
never "globally free" - the only ways another user can receive it are
(a) a genuine reclaim under the established sustained-utilization
policy, or (b) a resolved reallocation ask under this hysteresis
policy. Both always go through the one real confirmation flow;
neither is bypassed here.

**Decision trace.** `RoutingDecision` (unchanged) already lists every
GPU considered, utilization, and availability for ordinary routing.
Day 8 adds `LoadBalancingTrace`/`LoadBalancingCandidate`
(`engine/balancing/decision.py`) for the reallocation-ask path:
per candidate, utilization, holder, eligibility, `in_cooldown`, and -
when ineligible - exactly why (`"not underutilized enough"`,
`"priority not outranked"`, `"in cooldown"`, `"already declined for
this job"`, …). `Scheduler.last_balancing_traces` holds every trace
built by the most recent call, purely observational - it changes no
selection logic, only makes the existing one explainable. Events
(`BALANCE` for routing, `REQUEST`/`PRIORITY_PREEMPTION` for asks,
`ALLOC`/`STATUS` for the resulting placement) are unchanged.

**Design decision preserved, not changed (per the brief's own
caution):** an idle-but-still-assigned GPU is *never* treated as
routable new-work capacity by `route_job`. The only way it becomes
available to someone else is the reallocation-ask flow above, which
always asks the current holder first. Nothing here weakens that
ownership boundary.

## Priority Preemption & Wait-Time Aging

**Preemption asks the same question every other reclamation trigger
asks** (`ReclamationEngine.request_gpu_for_reallocation` → the one
confirm/respond/timeout state machine), just from a different reason:
*"another waiting job's claim on this GPU is stronger than its current
holder's."* `Scheduler._request_additional_gpus_if_needed`'s priority
path (unchanged from Day 8) evaluates it in two steps, both required:

1. **Tier gate** - the requester's `Priority` must *strictly* exceed
   the holder's. Equal tiers are ordinary allocation competition
   (Phase 3's territory), never preemption of an already-running job -
   this is what stops `HIGH > LOW` alone from ever being the reason.
2. **Score gate** - `AllocationEngine.calculate_score`, the same
   0.6×priority + 0.4×size + aging formula every allocation decision
   uses, computed head-to-head between the requester and the holder's
   job. Only a requester that *out-scores* the holder, not merely
   out-ranks it, is eligible - the blended-score policy governs
   preemption exactly like it governs everything else.

Both gates apply identically whether the requester needs 1 GPU or
several - there is no separate, gpu_count-gated code path for
single-GPU preemption.

**Two real bugs, both reproduced before fixing:**

- **Preempted jobs used to vanish.** `_reclaim` (the one function
  every reclamation/preemption/resource-request trigger already
  shared) always ended a fully-taken job at `JobStatus.RECLAIMED` -
  correct for a sustained-idle reclaim, wrong for preemption, where
  the holder plainly still needs a GPU. It now requeues a preempted
  job instead (`JobStatus.WAITING`, via the already-idempotent
  `requeue_job`) - and resets `submitted_at`/`started_at` to the
  requeue moment, so it competes fresh rather than instantly winning
  its old GPU back off the job that just preempted it under FCFS
  (verified directly: `test_preempted_job_does_not_immediately_win_
  its_old_gpu_back`).
- **Aging was inert for preemption twice over.** `calculate_score`
  was called without `now` (defaulting to zero wait), so a waiting
  job's aging never counted toward preempting anyone. Fixing that
  exposed a second issue: `calculate_allocation_score` measured
  elapsed time from `submitted_at` unconditionally, so once a
  *running* holder's own score started being computed for the first
  time (to compare against a requester's), its score kept "aging"
  forever too, right alongside the requester's - aging now freezes at
  `started_at - submitted_at` once a job has started, exactly like
  `Job.waiting_time` already does. Both fixes are corrective, not new
  policy - no formula, weight, or threshold changed.
- **A cancelled requester's ask could outlive it.** `Scheduler.
  cancel_job` withdrew the job but never called the already-existing
  `ReclamationEngine.cancel_pending_requests_for_job`, so a holder
  could still be asked to release a GPU for a job that no longer
  needed it. `cancel_job` now calls it.

**Partial preemption** falls out of the two existing paths acting
together, per waiting job, on GPUs individually: the excess-capacity
path (Day 8) covers underutilized GPUs first; only a genuine remaining
deficit reaches the priority-preemption gates above, so a holder's
actively-busy GPUs are asked for only when truly necessary, and a
holder always keeps whatever the requester's need didn't reach.

**Aging** - unchanged formula, now genuinely reachable everywhere it
should be:

| Parameter | Value | Meaning |
|---|---|---|
| Aging interval | continuous (function of elapsed minutes, not a discrete tick) | Recomputed fresh at every decision from `now - submitted_at` (or the frozen wait once running) |
| Aging increment | `AGING_RATE_PER_MINUTE = 0.01` | Score gained per minute waited |
| Maximum contribution | `AGING_MAX_CONTRIBUTION = 1.1` | Hard cap - `min(waiting_minutes * rate, cap)` |
| Can exceed base score? | Yes, deliberately | The cap (1.1) sits *above* the formula's own maximum base score (1.0 = `PRIORITY_WEIGHT + SIZE_WEIGHT`), so a job waiting `1.1 / 0.01 = 110` minutes always outscores any possible fresh arrival - a real guarantee, not an approximation |
| Simulated time | `now` is always caller-supplied (the `Scheduler`'s own simulated/test clock) | No wall-clock reads; every test in `test_priority_preemption_aging.py` drives it with explicit, deterministic timestamps |

**Starvation prevention, actually executed, not just inspected:**
`test_long_waiting_job_eventually_overtakes_a_continuous_stream_of_
newer_smaller_jobs` runs a real multi-round simulation - a single GPU,
a fresh small job re-contesting it every 5 simulated minutes - until
aging lets the one large, low-priority job originally submitted win a
round outright, and asserts it actually happens within a bounded
number of rounds.

**Preemption vs. automatic reclamation stay distinct**, exactly as
before: both end in a `RECLAIM` event through the same `_reclaim`, but
`metadata["category"]` is `PRIORITY_PREEMPTION` for one and
`AUTOMATIC_RECLAIM` for the other - never the same value, never a
second event type invented to tell them apart. The affected user's
own notification is that same backend `RECLAIM` event's `reason`
("your GPU allocation is being reclaimed for a higher-priority
scheduling request from …") - the exact event an admin's global feed
also sees; there is no separate, frontend-only notification.

**DSA roles, unchanged, reused exactly as documented elsewhere in this
file:** Priority Queue/Max-Heap orders waiting jobs by score for
ordinary allocation; the Queue preserves arrival order for FCFS ties
and is what a requeued job re-enters; hash maps resolve user/job/GPU
ids throughout the preemption flow; the Min-Heap remains the router's
GPU-selection structure (untouched here); the Sliding Window remains
sustained-utilization reclamation's own mechanism, never reused for
preemption's very different, score-based question. No new structure
was introduced.

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
