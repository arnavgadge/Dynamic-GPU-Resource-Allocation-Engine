# Architecture & Development Log — Phase 1 through 10

> This is the detailed, phase-by-phase engineering log for the Dynamic
> GPU Resource Allocation Engine: what each phase added, why, the
> exact algorithms/data structures involved, complexity notes, and
> worked traces. For a short orientation (what the project is, how to
> run it, current status), see the root [`README.md`](../README.md).
> For a structure-by-structure DSA reference, see [`dsa.md`](dsa.md).
> For install/run/test steps, see [`setup.md`](setup.md).

# Dynamic GPU Resource Allocation Engine — Phase 1 through 10

Team Vicimus | DSA Course Project | VIT Pune

**Phase 1** defines the **domain model**: the vocabulary later phases
(DSA structures, scheduling algorithms, OS/GPU integration, the
React frontend) are built on top of.

**Phase 2** adds the **DSA foundation**: the Linked List, Min-Heap,
Max-Heap, Priority Queue, HashMap, Queue, Stack and Sliding Window
that the scheduler will use — each one wired to a specific job in
this project, not implemented as a generic syllabus exercise.

**Phase 3** adds the **Allocation Engine** — the first phase allowed
to make a decision. Given an available GPU and a set of waiting
jobs, it applies the project's fixed policy (allocation score, 20%
job-size similarity rule, FCFS, critical-job precedence) to pick a
winner, assigns the GPU, and logs the decision as an `Event`.

**Phase 4** adds the **Reclamation Engine** — detects a GPU that has
been genuinely idle (not just briefly quiet) for a sustained period,
and safely returns it to the shared pool through a confirm-then-
reclaim flow, never by immediately yanking it the moment a threshold
is crossed.

**Phase 5** adds **load balancing and dynamic task routing** —
answers "given a job that should be scheduled, which currently
*available* GPU should receive it?" so new work naturally drifts
toward whichever GPU is least busy, without ever touching a GPU
that's already legitimately in use, no matter how low its
utilization reads.

**Phase 6** adds the **Scenario Simulator** — a deterministic backend
environment that drives the real engines through simulated time,
without a GPU, NVML, real users, or real OS processes. It generates
*inputs* (GPUs, users, jobs, utilization readings, user responses,
simulated time passing); it never generates a *decision*.

    The simulator generates inputs; the scheduler generates decisions.

Phases 1 and 2 contain **no scheduling logic** at all. Phase 3 is
scoped to *allocation* only. Phase 4 is scoped to *reclamation* only.
Phase 5 is scoped to *routing new work*, never migrating or
interrupting existing work. Phase 6 is scoped to *driving the other
five phases deterministically* - it contains no scheduling logic of
its own either.

**Phase 7** adds the **React dashboard and its backend API adapter**
- a terminal-styled control console that visualizes the real
`SchedulerState`/`Simulator` and sends control commands back, over a
small FastAPI layer that sits *above* `engine/` and contains no
scheduling logic of its own either.

    No scheduling logic exists in React. React renders decisions; Python makes them.

**Phase 8, Part A** adds the **hardware abstraction** - `GPUMonitor`,
one interface with three implementations (`SimulatorGPUMonitor`,
`NvidiaSMIMonitor`, `NVMLMonitor`) and a `MonitorPoller` bridge into
`Scheduler.record_utilization`, so the scheduler's decisions are
proven identical whether a reading came from the Phase 6 simulator or
real GPU hardware. (The rest of Phase 8's brief was cut off before it
reached this README - see that section for exactly what is and isn't
covered.)

No leases anywhere in any phase.

## Why this project exists

Companies allocate GPUs from a shared pool, but *allocated* capacity
often doesn't match *utilized* capacity — a user holding two GPUs
might only be actively using one, while someone else waits for
capacity. The engine watches utilization and reallocates freed
capacity, guided by rules and an allocation-score formula defined in
the full project report (`gpu_scheduler_full_report.docx`). None of
that decision logic is implemented yet — Phase 1 only makes sure the
data needed to implement it later actually exists, in a shape later
phases can use without a rewrite.

## Running it

```
pip install pytest
python main.py          # sample state -> DSA -> Phase 3 -> 4 -> 5, each traced -> Phase 6's full_lifecycle scenario
python -m pytest -v     # runs every backend unit/integration test (engine/ + api/)
```

### Running the full stack (backend + dashboard)

```
# Terminal 1 - backend (FastAPI + the real scheduler engines)
pip install fastapi uvicorn[standard] httpx websockets
python -m uvicorn api.app:app --reload --port 8000

# Terminal 2 - frontend (React + Vite dashboard)
cd frontend
npm install
npm run dev              # opens on http://localhost:5173, talking to the backend on :8000
```

Then open `http://localhost:5173`. The scenario selector defaults to
`gta5_excel`; pick `idle_user` or `full_lifecycle` from the dropdown
and press **START** to watch the complete allocation -> reclamation
-> reallocation lifecycle happen live, exactly as `main.py` prints it
in the terminal.

```
cd frontend && npm test    # 28 frontend tests (vitest + Testing Library)
cd frontend && npm run build  # production build check
```

## Project structure

```
engine/
  models/
    enums.py             GPUStatus, JobStatus, Priority, EventType, EngineStatus
    utilization.py        UtilizationObservation
    gpu.py                 GPU
    user.py                 User
    job.py                   Job
    assignment.py             GPUAssignment
    event.py                   Event
    scheduler_state.py           SchedulerState
  sample_data.py          builds the sample 4-GPU / 3-user state
  dsa/
    linked_list.py         LinkedList[T]              (generic)
    gpu_pool.py             GPUPool                     (wraps LinkedList[GPU])
    min_heap.py              MinHeap[T]                  (generic)
    gpu_utilization_heap.py    GPUUtilizationHeap          (wraps MinHeap[GPU])
    max_heap.py               MaxHeap[T]                  (generic)
    priority_queue.py           PriorityQueue[T]            (wraps MaxHeap[T])
    hashmap.py                   HashMap[K, V]                (generic)
    user_gpu_index.py             UserGPUIndex                 (wraps HashMap)
    queue.py                       Queue[T]                     (generic)
    waiting_job_queue.py             WaitingJobQueue               (wraps Queue[Job])
    stack.py                          Stack[T]                       (generic)
    reclaim_history.py                 ReclaimHistory                  (wraps Stack[Event])
    sliding_window.py                   UtilizationSlidingWindow          (built on Queue)
  allocation/
    config.py                             PRIORITY_WEIGHT, SIZE_WEIGHT, JOB_SIZE_SIMILARITY_THRESHOLD
    scoring.py                              normalized_priority, job_size_inverse, allocation_score
    similarity.py                             relative_size_spread, sizes_are_similar
    decision.py                                AllocationPolicy, CandidateInfo, AllocationDecision
    engine.py                                    AllocationEngine
  reclamation/
    policy.py                                      ReclamationTier(Policy), ConfirmationResponse, DEFAULT_RECLAMATION_POLICY
    monitor.py                                        is_sustained_breach
    decision.py                                          ReclamationAction, ReclamationDecision
    engine.py                                              ReclamationEngine
  balancing/
    config.py                                                BalancingPolicy, DEFAULT_BALANCING_POLICY
    availability.py                                            is_gpu_available, has_valid_utilization
    detector.py                                                  utilization_spread, is_pool_imbalanced
    decision.py                                                    RoutingOutcome, GPUCandidateInfo, RoutingDecision
    router.py                                                        LoadBalancingRouter
  scheduler.py            Scheduler - orchestrates Allocation + Balancing + Reclamation (no policy of its own)
  simulation/
    clock.py                                                          SimulationClock
    actions.py                                                          AddJobAction, UtilizationAction,
                                                                           UserResponseAction, CompleteJobAction
    scenario.py                                                           Scenario
    registry.py                                                           ScenarioRegistry, ScenarioInfo
    scenarios.py                                                          the 6 built-in scenario builders
    simulator.py                                                          Simulator
  hardware/
    metrics.py                                                              GPUMetrics, GPUProcessInfo
    monitor.py                                                                GPUMonitor (interface), MonitorUnavailableError
    simulator_monitor.py                                                       SimulatorGPUMonitor
    nvidia_smi_monitor.py                                                       NvidiaSMIMonitor
    nvml_monitor.py                                                              NVMLMonitor
    factory.py                                                                    detect_gpu_monitor
    poller.py                                                                      feed_metrics, MonitorPoller
api/                    the Backend/API Adapter (Phase 7) - imports engine/, engine/ never imports this
  config.py                 ALLOWED_SPEEDS, BASE_TICK, TICK_INTERVAL_SECONDS - control-surface config only
  session.py                 SimulationSession - the one live Simulator + running/speed state
  serializers.py               dataclass/enum -> plain JSON dicts (the only place that happens)
  schemas.py                    pydantic request bodies (speed, step, prompt response)
  app.py                          FastAPI app - every REST route + the /ws/state WebSocket
tests/                    unit tests for every model above
  dsa/                    unit + integration tests for every DSA structure above
  allocation/             unit tests (scoring, similarity) + scenario/integration tests for the engine
  reclamation/            unit tests (sustained-breach detection) + full confirm/reclaim flow tests
  balancing/              unit tests (availability, imbalance) + 15 routing scenarios + 2 full integration tests
  simulation/             clock, registry, simulator mechanics, reclamation-through-simulated-time,
                            determinism (every scenario run twice), and the full end-to-end scenario
  api/                    FastAPI route tests - scenarios, controls, prompt response, WebSocket, validation
  hardware/               each monitor in isolation (mocked subprocess/pynvml), the factory's probe logic,
                            the poller bridge, and the source-agnostic equivalence test
frontend/                 the React dashboard (Phase 7) - Vite + plain CSS, no UI framework
  src/
    components/              Header, SystemOverview, EngineStatus, UsersPanel, GPUPool, GPURow,
                                UtilizationBar, StatusBadge, WaitingQueue, EventLog, EventEntry,
                                ScenarioSelector, SimulationControls, ReclamationPrompt, Footer
    hooks/useSchedulerState.js   the one coherent source of frontend scheduler state
    services/                     api.js (REST), websocket.js (live state stream) - the only fetch/WebSocket calls
    utils/                         formatters.js, eventStyles.js - pure display formatting, no logic
    App.jsx, main.jsx, index.css
  src/__tests__/            component + service tests (vitest + Testing Library)
main.py                    demo script: sample state -> DSA -> allocation -> reclamation -> routing -> simulator
```

## The models

**`GPU`** — one physical GPU in the company pool: id, memory
(total/used), current utilization, `status` (a `GPUStatus`), who it
is currently assigned to (`assigned_user_id` / `assigned_job_id`,
both optional — a GPU may be unassigned), and a history of
`UtilizationObservation`s. `record_observation()` only appends data
and syncs the "current" fields — it never changes `status` or
assignment. Deciding what a low reading *means* is scheduler logic.

**`User`** — an employee: id, name, `priority` (a `Priority`), the
GPUs they currently hold (`assigned_gpu_ids`, zero to many — no
GPU is permanently owned), and their running jobs. A user never
picks their own GPU.

**`Job`** — a submitted workload: id, owning user, a free-text
`name` (`"ML Training"`, `"Excel"`, `"GTA5"`, …) used only for
display, `priority`, `estimated_size_minutes`, `status` (a
`JobStatus`), timestamps, and `assigned_gpu_id`. Nothing in the model
branches on `name` — the engine is workload-agnostic by construction,
not by convention.

**`GPUAssignment`** — a small, explicit record: "GPU X was given to
user Y (running job Z) at time T, and ended at time T2 (or is still
active)." See *Design decisions* below for why this exists alongside
the pointer fields on GPU/User/Job.

**`Event`** — one entry in the real-time event log: id, timestamp,
`event_type` (an `EventType`), optional related GPU/user/job, a
`message`, and an optional `reason`. Purely descriptive — no event
processing or generation logic lives here.

**`SchedulerState`** — the central container: dictionaries of all
GPUs/users/jobs/assignments, the event log, and `engine_status`. It
also exposes read-only query helpers (`get_gpus_for_user`,
`get_job_on_gpu`, `get_waiting_jobs`, `get_active_assignment_for_gpu`,
`get_events_for_gpu`, …) that answer the relationship questions the
project spec calls out. These are filters over plain dicts/lists —
not scheduling decisions, and not yet the HashMap/heap
implementations Phase 2 will introduce.

## Enums

- **`GPUStatus`**: `ACTIVE`, `IDLE`, `IDLE_WARNING`, `RECLAIMING`,
  `REALLOCATING` — the GPU's current lifecycle state, not a decision
  it made about itself.
- **`JobStatus`**: `WAITING`, `RUNNING`, `COMPLETED`, `RECLAIMED`.
  Kept intentionally small — a separate "interrupted" state was
  considered and dropped, since at this stage both cases mean the
  same thing: the job stopped running before completion.
- **`Priority`** (`IntEnum`): `LOW=1`, `MEDIUM=2`, `HIGH=3`,
  `CRITICAL=4`. It's an `IntEnum`, not a plain label, so the later
  allocation-score formula (`0.6 × priority + 0.4 × size_inverse`,
  per the project report) can use the value directly in arithmetic.
  `CRITICAL` sits above `HIGH` because the report calls out critical
  jobs that must be scored ahead of everything else. Priority is
  always an *external* input — nothing in this codebase computes it.
- **`EventType`**: `SYSTEM`, `ALLOC`, `MONITOR`, `PROMPT`, `RESPONSE`,
  `RECLAIM`, `STATUS`, `REQUEST`, `BALANCE`.
- **`EngineStatus`**: `STOPPED`, `RUNNING`, `PAUSED` — the scheduler
  engine's own run state, held on `SchedulerState`.

## Design decisions worth calling out

**Why a separate `GPUAssignment` model, instead of only fields on
GPU/User/Job?** The pointer fields (`GPU.assigned_user_id`,
`User.assigned_gpu_ids`, `Job.assigned_gpu_id`) are enough to answer
"what's true *right now*" in O(1) — that's what Phase 2's algorithms
will read. But the spec also needs "when was this assignment
created" and "when did it change", and later phases need an audit
trail to reason about reclaim/reallocation history. Keeping
timestamps in sync across three different objects is error-prone, so
`GPUAssignment` carries `created_at` / `ended_at` as its own small,
explicit record. GPU/User/Job stay fast "current state" pointers;
`GPUAssignment` is the history those pointers are drawn from.

**Why no lease field anywhere.** The report explicitly rules out
leases: no GPU belongs to anyone for a fixed period, and the
scheduler can end an assignment (`GPUAssignment.end()`) for any
reason a later phase decides. There is no expiry timer in the model.

**Why `dict`/`list` inside `SchedulerState` right now.** The
project's later DSA phase introduces HashMap-style O(1) lookup,
Min/Max-Heaps, and a Priority Queue as *justified*, purpose-built
structures (idle-GPU detection, allocation-score ordering, waiting-job
management). Reimplementing hand-rolled versions of those in Phase 1,
before there is any algorithm to feed them, would be scaffolding
without a job to do. Python's built-in `dict` already gives O(1)
lookup for the "current state" queries Phase 1 needs to prove out;
Phase 2 swaps in the specialized structures where the algorithms
actually need their properties (e.g. ordering by allocation score).

**Why utilization history lives on the `GPU`, not in a separate
time-series store.** Later phases need to scan a *single GPU's* past
observations to detect sustained thresholds (e.g. "below 2% for
20–30 minutes"). Keeping the list on the `GPU` object keeps that scan
local and simple; a sliding-window algorithm over
`gpu.utilization_history` is a Phase 2 concern, not a Phase 1 one.

## Sample state

`engine/sample_data.py` builds the scenario from the project report:
4 GPUs, 3 users (Alice/HIGH, Bob/MEDIUM, Charlie/LOW), each of the
first four jobs running on its own GPU, plus a fifth job for Charlie
(`"Video Editing"`) left `WAITING` to exercise the waiting-job path
without needing a fourth user. It is demo/test data only — the real
system must handle an arbitrary number of GPUs, users and jobs, and
this data is never read by anything that makes decisions.

## DSA Architecture (Phase 2)

Every structure below exists because a specific piece of the future
scheduler needs it — not to check a syllabus box. Two are used
exactly as the generic textbook structure (`HashMap`, `LinkedList`
via `GPUPool`); the others are generic building blocks with a thin,
project-specific wrapper on top so the underlying DSA component stays
reusable while the wrapper carries the project's vocabulary.

| Data Structure | Purpose in this project | Complexity |
|---|---|---|
| **Linked List** (`GPUPool`) | The company's GPU inventory. A linked list gives O(1) insertion/removal as GPUs are dynamically added to or removed from the pool — no shifting elements the way a fixed array would need. | append/prepend O(1); find/remove O(n); size O(1) |
| **Min-Heap** (`GPUUtilizationHeap`) | Efficiently answers "which GPU is least utilized right now?" — the signal the future idle/underutilized-GPU detector will read. Does not decide whether that GPU should be reclaimed. | insert O(log n); extract-min O(log n); peek O(1); rebuild (after external mutation) O(n) |
| **Max-Heap** | The generic ordering engine `PriorityQueue` is built on. Kept separate and comparator-driven so no priority *rule* is baked into the heap itself. | insert O(log n); extract-max O(log n); peek O(1); rebuild O(n) |
| **Priority Queue** | Manages waiting jobs by whatever comparator the scheduler eventually supplies (the allocation-score formula, in Phase 3) — built on the Max-Heap above, with insertion-order tie-breaking for equal priorities. | insert O(log n); pop-best O(log n); peek O(1) |
| **HashMap** (`UserGPUIndex`) | Fast user → currently-assigned-GPU(s) lookup, built from `GPUAssignment` records. A user can hold zero, one, or many GPUs, so each key maps to a list, never a single value. | put/get/remove/contains O(1) average (amortized resize); O(n) worst case |
| **Queue** (`WaitingJobQueue`) | Preserves arrival order for waiting jobs, ready for when the scheduler's 20%-similarity rule (Phase 3) decides two jobs should be served FCFS instead of by score. | enqueue/dequeue/peek O(1) |
| **Stack** (`ReclaimHistory`) | Records reclaim `Event`s so the most recent reclaim is always what a future rollback would undo first. | push/pop/peek O(1) |
| **Sliding Window** (`UtilizationSlidingWindow`) | Keeps only the `UtilizationObservation`s inside a trailing time window, so the future reclamation engine can tell a *sustained* low reading (Rule 1's 20–30 min / 2–3 hr tiers) apart from a brief dip, without rescanning a GPU's entire history each time. | add O(1) + amortized O(1) eviction per observation; window query O(n) in current window; span O(1) |

**Why a Max-Heap *and* a Priority Queue, when they sound like the
same thing?** They're not duplicated — they're layered. `MaxHeap[T]`
is the raw, comparator-free array-heap mechanism (sift-up/sift-down).
`PriorityQueue[T]` is that mechanism wrapped with a caller-supplied
`key_fn` and FIFO tie-breaking, which is what the project report's
"Priority Queue with custom comparator" central data structure
actually is. Building the Priority Queue *on top of* the Max-Heap,
rather than writing two independent heaps, is what keeps the Max-Heap
itself free of any hardcoded priority rule.

**Why the GPU pool is a real linked list and not `list.append`.**
Python's `list` already gives amortized O(1) append, so the
difference isn't raw speed — it's that a `list` doesn't teach or
demonstrate anything about node-based structures, and the report
explicitly calls for a linked list here. `LinkedList` is implemented
with actual node objects and head/tail pointers; `GPUPool` is the
thin, GPU-flavoured API around it.

**Why the HashMap is hand-built instead of `dict`.** Same reasoning:
`dict` would work, but wrapping it teaches nothing. `HashMap` is a
real bucket array with separate-chaining collision handling and
doubles its capacity past a 0.75 load factor, same as production
hash tables do it.

**Why `Queue` uses head/tail node pointers but `Stack` uses a plain
list.** A queue must add at one end and remove from the other;
`list.pop(0)` for that would be O(n) because every remaining element
shifts down, so `Queue` needs real linked nodes for O(1) on both
ends. A stack only ever touches one end, which is exactly what
`list.append`/`list.pop()` (no index argument) already do in O(1) —
adding node-linking there would add complexity without changing the
complexity class.

## Allocation Engine (Phase 3)

The `AllocationEngine` (`engine/allocation/engine.py`) answers one
question: *a GPU is available, several jobs are waiting - which job
gets it?* It follows one fixed pipeline, in this order, every time:

```
GPU becomes available
        v
Check waiting jobs                         (no candidates -> put the GPU back, stop)
        v
Check critical jobs                        (any CRITICAL job present -> only critical jobs compete this round)
        v
Compute the size spread of the candidates
        v
Spread <= 20%?  --yes-->  FCFS              (earliest Job.submitted_at wins)
        |
        no
        v
Score-based selection                       (0.6 x priority + 0.4 x job-size-inverse, highest wins)
        v
Assign GPU, update state, log an ALLOC event, return the decision
```

### 1. Allocation score

```
Allocation Score = (0.6 x normalized priority) + (0.4 x job size inverse)
```

Implemented in `engine/allocation/scoring.py`, with the 0.6/0.4
weights defined exactly once in `engine/allocation/config.py`
(`PRIORITY_WEIGHT`, `SIZE_WEIGHT`) and asserted to sum to 1.0.

### 2. Why priority is 60%

Priority protects important company work - the project's own
reasoning is that a missed deadline is worse than briefly
inefficient GPU usage, so priority outweighs the size term.

### 3. Why job size is 40%

Job size still matters because a smaller job finishes sooner and
frees the GPU for whoever is waiting next - the project deliberately
keeps this a real factor, not a tie-breaker, hence 40% rather than a
token weight.

### 4. Job-size inverse - the normalization that keeps it safe

`job_size_inverse` does **not** compute `1 / size` (Section 6/23 of
the brief explicitly warns against this - it would blow up for a
zero-length job and would depend on whatever unit "size" is measured
in). Instead it min-max normalizes **relative to the current
candidates**: the smallest candidate scores `1.0`, the largest scores
`0.0`, and everything else falls linearly in between. A pathological
size (zero, negative) is clamped into that range instead of causing
an error. If every candidate is the same size, everyone scores `1.0`
- there is nothing to differentiate on, so no one is penalized for
being "the largest" among equals.

Priority, by contrast, is normalized against **fixed** bounds
(`Priority.LOW` .. `Priority.CRITICAL`, i.e. 1..4) rather than the
current candidates - priority already has a known, global scale, so
its meaning shouldn't drift depending on who else happens to be
waiting right now. This asymmetry (fixed bounds for priority,
relative bounds for size) is the key design decision in `scoring.py`
and is documented there in detail.

Neither normalization can ever divide by zero, produce `NaN`, or
produce `inf` - `tests/allocation/test_scoring.py` exercises zero,
negative, and equal-min/max sizes directly.

### 5. The 20% job-size similarity rule

One constant, `JOB_SIZE_SIMILARITY_THRESHOLD = 0.20` in
`engine/allocation/config.py`, used in exactly one place
(`engine/allocation/similarity.py`). The metric is the **relative
size spread**: `(largest - smallest) / largest` over the candidate
set. Spread `<= 20%` -> FCFS; otherwise -> score-based selection.

```
10 / 12 / 11 minutes -> spread = (12-10)/12 = 16.7%   -> similar -> FCFS
15 min / 1 hr / 3 hr  -> spread = (180-15)/180 = 91.7% -> not similar -> score-based
```

**A documentation note, not a bug:** the project brief also gives
"15 / 10 / 13 minutes -> within 20%" as an example. Under this (or
any single reasonable) ratio metric that triple's spread is actually
`(15-10)/15 ≈ 33%` - outside 20%. The brief's two examples read as
"about the same size" vs. "different orders of magnitude"
illustratively, and are not simultaneously satisfiable by one precise
formula. Rather than special-case the threshold to reproduce one
example (which the brief explicitly warns against - "do not interpret
20% as an arbitrary constant in several different places"), this
implementation uses the single definition above everywhere, and its
tests use the brief's other, consistent example (10/12/11 vs.
15/60/180).

### 6. FCFS behavior

When candidates are similar in size, the winner is whichever has the
smallest `Job.submitted_at` - the actual submission timestamp Phase 1
already records, not queue position, not job id, not name. The
`WaitingJobQueue` (Phase 2's FCFS `Queue` wrapper) is still what
`AllocationEngine` stores every waiting job in and removes them from
- it is the actual source of truth for "who is currently waiting" -
but the tie-break itself reads `Job.submitted_at` directly, so a
decision stays correct even if jobs were ever submitted out of strict
chronological call-order (bulk import, tests, a restarted engine
replaying persisted jobs).

### 7. SJF / score-based behavior

When sizes clearly differ, every candidate's `allocation_score` is
computed against the same `(min_size, max_size)` from that candidate
set, and the highest score wins (ties broken by earliest
`submitted_at`). This is **not** "always pick the smallest job" -
`tests/allocation/test_scoring.py::test_not_pure_sjf_smallest_job_does_not_always_win`
and `test_priority_can_outweigh_a_large_size_disadvantage` both
demonstrate a higher-priority, larger job outscoring a low-priority,
tiny one.

### 8. Critical-job handling

Integrated directly into `select_next_job`, not a separate scheduler:

```python
critical_candidates = [job for job in candidates if job.priority == Priority.CRITICAL]
pool = critical_candidates if critical_candidates else candidates
```

If any `CRITICAL` job is waiting, the entire rest of the pipeline
(similarity check, FCFS-or-score, the winner) runs on **only** the
critical jobs - normal-priority jobs are not compared against them at
all this round, and stay waiting for a future round once no critical
job remains. If several critical jobs are waiting, the same
similarity/FCFS/score logic still applies *among* them - critical
does not mean "skip the policy", it means "shrink the candidate pool
to critical jobs first".

### 9. Allocation flow end to end (`AllocationEngine`)

`engine/allocation/engine.py`:

- `add_gpu` / `remove_gpu` / `mark_gpu_available` manage the GPU pool
  and which GPUs are currently available.
- `submit_job` registers a job and enqueues it if `WAITING`; rejects
  a duplicate id or a non-`WAITING` job outright (fails clearly
  instead of silently misbehaving).
- `select_next_job` is pure decision logic - critical-tier gate,
  similarity check, FCFS-or-score - and mutates nothing.
- `allocate_next` pops one available GPU, calls `select_next_job`,
  and (if there was a candidate) commits the decision: moves the job
  to `RUNNING`, the GPU to `ACTIVE`, records a `GPUAssignment`,
  updates the `User`'s `assigned_gpu_ids`/`running_job_ids`, updates
  the `UserGPUIndex`, and logs an `ALLOC` `Event`. If there was no
  candidate, the GPU is put back as available rather than lost.
- `allocate_all` repeats `allocate_next` until no GPU or no job is
  left - Scenario 7's "drain the queue" behavior.

No GPU lease is ever created, and no GPU is treated as permanently
owned by a user - `is_assigned` is always the live, current truth.

### 10. How the Phase 2 DSA structures are actually used

| Structure | Role in `AllocationEngine` |
|---|---|
| `GPUPool` (linked list) | The full GPU inventory - `add_gpu`/`remove_gpu`/lookup. |
| `GPUUtilizationHeap` (min-heap) | Holds **only currently-available** GPUs; `allocate_next` extracts the least-utilized one. Never confused with "least utilized among all GPUs" - a busy GPU is never in this heap. |
| `WaitingJobQueue` (Queue) | The actual storage for every waiting job, in arrival order; candidates for a decision are read from it, and the winner is removed from it via a drain-and-rebuild (documented in `_remove_from_waiting_queue`, the same technique `MinHeap.rebuild` uses). |
| `UserGPUIndex` (hashmap) | Updated on every assignment; `get_gpus_for_user` is an O(1) lookup of what a user currently holds. |

`MaxHeap`/`PriorityQueue` from Phase 2 remain available (and are
still exercised in `main.py`'s Phase 2 demo with a placeholder
comparator) but `AllocationEngine` itself sorts the small, per-decision
candidate list directly (`min`/`max` over at most "however many jobs
are waiting", not the whole system) - see Complexity below for why
that is not "bypassing the DSA layer".

### 11. Complexity

| Operation | Complexity | Why |
|---|---|---|
| `add_gpu` / `mark_gpu_available` | O(1) append + O(log n) heap insert | `GPUPool.add_gpu` is O(1); inserting into the available-GPU min-heap is O(log n). |
| `remove_gpu` | O(n) | `GPUPool.remove_gpu` scans the linked list (Phase 2's documented linked-list removal cost). |
| `_pop_available_gpu` | O(log n) amortized | One `extract_least_utilized` per allocation; occasional extra pops for lazily-discarded stale entries don't change the amortized bound. |
| `submit_job` | O(1) | Dict membership check + `WaitingJobQueue.enqueue_job`. |
| `select_next_job` | O(k) | k = candidates in the relevant tier (critical-only, or all waiting). Computing sizes, spread, and either the FCFS min or every candidate's score are all single O(k) passes - no per-candidate structure lookup grows this. |
| `_remove_from_waiting_queue` | O(n) | n = jobs currently waiting; drains and rebuilds the FIFO, same rationale as `MinHeap.rebuild`. |
| `get_gpus_for_user` (`UserGPUIndex`) | O(1) average | Direct `HashMap` lookup. |
| `allocate_next` | O(log n + k) | Sum of the above: one heap pop plus one O(k) selection pass. |
| `allocate_all` | O(m . (log n + k)) | m = number of allocations actually made. |

`k` (candidates considered) and `n` (GPUs/jobs tracked) are small in
this project's scale (a handful of GPUs, a waiting list of jobs) -
these bounds are what the implementation actually does, not
asymptotic claims beyond what the code performs.

### 12. Example allocation decision trace

From `main.py`'s `demo_allocation()` - one GPU, three jobs of
different priority and size:

```
GPU GPU0 became available. Candidates:

  J1   priority=HIGH     size=20 min  score=0.789
  J2   priority=MEDIUM   size=15 min  score=0.600
  J3   priority=LOW      size=200 min  score=0.000

Policy: SCORE_BASED
Winner: J1
Reason: no critical jobs waiting - scoring open to all waiting jobs;
        job sizes differ beyond 20% (spread=92.5%) -> score-based
        selection; J1 scored highest (0.789)
Event:  [ALLOC] Assigned GPU0 to Alice
```

J1 (HIGH, 20 min) beats J2 (MEDIUM, 15 min - the *smaller* job) because
priority is weighted 60% - a clean, reproducible demonstration that
this is neither pure SJF nor pure priority scheduling.

## Reclamation Engine (Phase 4)

The `ReclamationEngine` (`engine/reclamation/engine.py`) answers a
different question from Phase 3's: *has this assigned GPU been
genuinely idle long enough that it should be asked for back?* It
never reclaims the instant a threshold is crossed - only after
confirming the breach was **sustained**, and only after the user has
had a chance to say "still using it":

```
UTILIZATION THRESHOLD BREACH
        v
SUSTAINED CONDITION CONFIRMED     (is_sustained_breach, over the GPU's UtilizationSlidingWindow)
        v
IDLE_WARNING                      (GPU.status)
        v
PROMPT USER                       ("are you still using this GPU?", an Event)
        v
   +---------+----------+------------------+
   v         v          v
  YES        NO      NO RESPONSE
   v         v          v
Reset timer  Reclaim   Wait no_response_grace_period
/ back off  immediately        v
                           Auto reclaim
```

### 1. The two utilization tiers

Defined once in `engine/reclamation/policy.py`, never as inline magic
numbers:

| Tier | Threshold | Sustained duration | Interpretation |
|---|---|---|---|
| Tier 1 | < 2% | **25 minutes** (default; brief specifies 20-30) | Job likely completed |
| Tier 2 | < 15% | **2 hours 30 minutes** (default; brief specifies 2-3 hr) | User likely inactive |

Both defaults sit at the midpoint of the range the brief gives, and
both are constructor arguments (`ReclamationTierPolicy.sustained_duration`,
part of the `ReclamationPolicy` passed into `ReclamationEngine`) - a
demo can pass a policy with 25-*second* durations instead of minutes
without touching any engine code, which is exactly what
`tests/reclamation/test_engine.py`'s `TEST_POLICY` does.

Tier 1 is checked before Tier 2 (`ReclamationEngine._detect_tier`):
anything quiet enough to breach Tier 1's 2% threshold is, by
definition, also under Tier 2's higher 15% threshold, so if Tier 1's
(shorter) window has already confirmed a sustained breach there is no
reason to wait out Tier 2's much longer one as well.

### 2. Sustained, not momentary - the `is_sustained_breach` check

`engine/reclamation/monitor.py` is deliberately *not*:

```python
if utilization < 15:
    reclaim()
```

It reasons over history: given a GPU's recent `UtilizationObservation`s
(oldest first), the breach is "sustained" only if **every** reading
inside the trailing window `[latest - duration, latest]` is below the
threshold, and enough history exists to actually prove the full
duration elapsed low (the earliest available observation must be at
or before that window's start - a gap in monitoring data is never
mistaken for a real sustained-low reading). A single high reading
anywhere in that window - even briefly, even in the middle - breaks
the streak:

```
10%, 9%, 4%, 1%, 8%, 14%, ...   -> never dips low long enough -> not sustained
1%, 1%, 1%, 1%, ... (25 min)     -> continuously below 2% for the full window -> sustained
```

`tests/reclamation/test_monitor.py` exercises this directly: brief
dips that recover, insufficient history, a single boundary reading
exactly at the threshold, and a spike that happened *before* the
required window (and therefore doesn't matter).

### 3. Where the Sliding Window (Phase 2) fits in

Each GPU being watched gets exactly one `UtilizationSlidingWindow`,
sized to `ReclamationPolicy.monitoring_window_duration` - the
*larger* of the two tiers' durations (Tier 2's, by default). One
window this size covers both tiers' checks, so a GPU's observations
are never stored twice. `record_utilization` feeds every new reading
into both the GPU's own history (Phase 1's `GPU.record_observation`)
and this window; `_detect_tier` reads `window.observations(now)` -
never the GPU's entire lifetime of history - to run the sustained
check above.

### 4. The confirmation flow, and why "reset timer" doesn't touch the window

- **Threshold breach confirmed** -> `_raise_prompt`: sets
  `GPU.status = IDLE_WARNING`, logs a `PROMPT` `Event`
  ("are you still using this GPU?"), and remembers `prompted_at` +
  which tier fired. While a prompt is outstanding, new utilization
  readings are recorded but never raise a *second* prompt.
- **YES** -> `respond(..., ConfirmationResponse.YES)`: logs a
  `RESPONSE` event, sets `GPU.status` back to `ACTIVE`, and records
  `confirmed_at = now`. "Reset the timer" does **not** mean clearing
  the sliding window (that's real monitoring data) - it means every
  future sustained-breach check for this GPU ignores observations at
  or before `confirmed_at`, so the GPU has to breach for the *entire*
  tier duration all over again before it can be prompted a second
  time.
- **NO** -> `respond(..., ConfirmationResponse.NO)`: reclaims
  immediately (see below).
- **No response** -> `check_timeouts(now)`: for any GPU with a prompt
  older than `no_response_grace_period` (default 5 minutes, also
  configurable), reclaims exactly the same way a "no" would. Call
  this periodically (or before consuming new utilization data) so a
  GPU is not left in `IDLE_WARNING` forever just because monitoring
  data stopped arriving once the job actually finished.

### 5. Reclaiming - keeping GPU/Job/User/Assignment consistent

`_reclaim` mirrors Phase 3's `_assign_gpu`, symmetrically undoing it:

- `Job.status -> RECLAIMED`, `Job.assigned_gpu_id -> None`.
- `User.assigned_gpu_ids` / `running_job_ids` have the GPU/job removed.
- The active `GPUAssignment` is ended (`GPUAssignment.end(now)`) -
  never deleted, so it stays in `SchedulerState.assignments` as
  history.
- `GPU.assigned_user_id` / `assigned_job_id -> None`, `GPU.status ->
  IDLE`.
- A `RECLAIM` `Event` is logged and pushed onto `ReclaimHistory`
  (Phase 2's `Stack` wrapper) - `ReclamationEngine.last_reclaim()` /
  `undo_last_reclaim()` expose it directly.
- If an `AllocationEngine` was given to the `ReclamationEngine`, the
  freed GPU is hand back via `mark_gpu_available` - the exact same
  pool every other GPU is allocated from. **No lease is ever created**
  and no GPU is ever "returned to its owner" - there was no owner,
  only a holder whose assignment just ended.

### 6. Complexity

| Operation | Complexity | Why |
|---|---|---|
| `record_utilization` | O(1) amortized + O(t) | `GPU.record_observation` and `window.add_observation` are O(1) amortized (Phase 1/2); `_detect_tier`'s scan of the window's current contents is O(w) where w = observations in the monitoring window, checked against t = 2 tiers - a small constant. |
| `is_sustained_breach` | O(w) | One linear pass over the observations currently in the window - no re-scanning of a GPU's entire lifetime history, which is exactly what the Sliding Window (Phase 2) exists to avoid. |
| `respond` / `_reclaim` | O(1) | Dict lookups and constant-size list membership checks/removals on a user's small `assigned_gpu_ids`/`running_job_ids`. |
| `check_timeouts` | O(g) | g = GPUs currently being watched - one grace-period check each. |
| `ReclaimHistory` push / peek / pop | O(1) | Phase 2's `Stack` wrapper, unchanged. |

### 7. Example trace

From `main.py`'s `demo_reclamation()` - a GPU sits at 1% utilization
for the full Tier 1 duration, is prompted, and gets no response:

```
Policy: Tier 1 < 2% sustained for 0:25:00; no-response grace period 0:05:00

After 25 minutes at 1% utilization:
  GPU status: IDLE_WARNING
  Decision: PROMPTED (TIER_1) - TIER_1 (likely completed job): utilization
            sustained below 2% for at least 0:25:00
  Event: [PROMPT] GPU0: are you still using this GPU?

No response for 0:05:00 + 1 minute:
  RECLAIMED: Reclaimed GPU0 from Alice (TIER_1 sustained breach - no response
             within the grace period)
  GPU status: IDLE
  Job status: RECLAIMED
  GPUs available for allocation again: 1
```

## Load Balancing Engine (Phase 5)

`LoadBalancingRouter` (`engine/balancing/router.py`) answers a third
question, distinct from both engines before it:

```
Phase 3 (AllocationEngine)    ->  WHICH JOB should be scheduled next?
Phase 4 (ReclamationEngine)   ->  WHEN should an idle GPU be taken back?
Phase 5 (LoadBalancingRouter) ->  WHICH GPU should a job that's being
                                    scheduled actually go to?
```

### 1. Purpose

When a new job needs a GPU, blindly taking "the first GPU in the
pool" (or, worse, whichever GPU merely *looks* idle by utilization)
risks either poor utilization or - far worse - stealing a GPU a user
is legitimately still using. The router's job is to notice when a
GPU that is genuinely free is sitting at low utilization while
another *available* GPU is busier, and prefer the free, quiet one -
without ever touching a GPU that is not actually free.

### 2. Four different concepts that must not be conflated

This distinction is the entire point of the phase:

| Concept | What it means | Where it's decided |
|---|---|---|
| **Utilization** | `GPU.utilization_percent` - a raw signal, nothing more. | Reported by monitoring (simulated so far). |
| **Assigned** | `GPU.is_assigned` - does *anyone* currently hold this GPU? | Set by whichever engine last committed an assignment/reclaim. |
| **Available** | Genuinely free to hand a *new* job right now. | `engine.balancing.availability.is_gpu_available`: `not is_assigned and status == IDLE`. |
| **Reclaimable** | A *judgment about an existing assignment*, made over sustained time. | Phase 4 only - this module never reads or computes it. |

A GPU at 4% utilization that is still `assigned` is **not**
available - no matter how idle it looks, it is not this phase's to
touch. It only becomes available once Phase 4 independently decides
to reclaim it and the GPU model reflects that
(`assigned_user_id is None`, `status == IDLE`) - Phase 5 never
shortcuts that by reading utilization directly.

`is_gpu_available` treats exactly one status as available: `IDLE`.
`ACTIVE` and `IDLE_WARNING` are excluded because the GPU is still
assigned; `RECLAIMING` and `REALLOCATING` are excluded even on an
already-unassigned GPU, because a state transition that hasn't
finished settling is not safe to route new work onto yet.

### 3. Routing logic

```python
def route_job(job, candidate_gpus=None):
    pool = candidate_gpus or every GPU in SchedulerState
    available = [gpu for gpu in pool if is_gpu_available(gpu)]
    routable = [gpu for gpu in available if has_valid_utilization(gpu)]
    if not routable:
        return NO_GPU_AVAILABLE   # job stays WAITING - never a fake allocation
    selected = min(routable, key=lambda gpu: (gpu.utilization_percent, gpu.gpu_id))
    return ROUTED(selected)
```

(The real implementation builds a `MinHeap` rather than calling
`min` directly - see the DSA section below; the logic is identical.)

`has_valid_utilization` guards against a monitoring glitch (`NaN`,
`inf`, a negative or >100 reading) ever winning a "lowest
utilization" comparison - such a GPU is excluded from selection
entirely rather than risk an unsafe routing decision (`NaN` compares
as neither less nor greater than anything, which would make ordering
behave inconsistently).

### 4. Min-Heap integration (Phase 2, reused)

Exactly the architecture the brief suggests:

```
candidate GPUs -> filter for availability (+ utilization safety) -> MinHeap by utilization -> extract_min
```

The filtering is a single O(n) pass over the pool *before* anything
touches the heap - the heap is never built from (and can never
accidentally return) a GPU that isn't already known to be available.
The key is the tuple `(utilization_percent, gpu_id)`: two GPUs tied
on utilization are broken by comparing `gpu_id` next, which is why
the tie-break is deterministic (lexicographic by id) rather than
whatever order a dict or an unstable sort happened to produce. No
second heap is built anywhere else for this purpose - the router
reuses `engine.dsa.min_heap.MinHeap` (the exact same generic
structure `GPUUtilizationHeap` wraps in `engine/allocation/`)
directly, since here the *filtering itself* is the domain-specific
step, not the heap.

### 5. HashMap and Linked List (Phase 2, reused)

- **HashMap**: after the heap picks the winning GPU, the router looks
  it back up via an `engine.dsa.hashmap.HashMap[str, GPU]` built over
  the available candidates for that call - the same O(1)-average
  assignment-lookup pattern `UserGPUIndex` already uses elsewhere in
  this project, applied here to "GPU id -> GPU object" instead of
  "user id -> GPU ids".
- **Linked List**: the router draws its candidate pool from
  `AllocationEngine.gpu_pool` - the exact same `GPUPool` (Phase 2's
  linked list) `AllocationEngine` already maintains as the company's
  full GPU inventory. Phase 5 does not create a second pool.

### 6. Interaction with the Allocation Engine - no duplicated scoring

The router never re-runs the allocation-score formula, never decides
which job goes next, and never picks between waiting jobs - all of
that stays exactly where Phase 3 left it
(`AllocationEngine.select_next_job`). The two engines meet at exactly
one new method, `AllocationEngine.finalize_assignment(job, gpu,
policy, reason, candidates)` - a one-line public wrapper around the
same private `_assign_gpu` Phase 3's own `allocate_next` already
used, so a routed assignment and a self-contained Phase-3-only one
are committed identically:

```python
job, policy, reason, candidates = allocation_engine.select_next_job()   # Phase 3: WHICH JOB
routing = router.route_job(job, allocation_engine.gpu_pool.all_gpus())   # Phase 5: WHICH GPU
gpu = state.get_gpu(routing.selected_gpu_id)
decision = allocation_engine.finalize_assignment(job, gpu, policy, reason, candidates)  # commit
```

`select_next_job` and `route_job` are both pure decision logic -
neither mutates anything - so trying either one twice without
committing is safe and returns the same answer (see
`test_14b_routing_without_committing_returns_the_same_gpu_again`).
Because the GPU the router chose already came from
`AllocationEngine.gpu_pool` (the same object `_pop_available_gpu`'s
stale-entry check already guards), a GPU that gets committed through
`finalize_assignment` is automatically skipped if `allocate_next`'s
own heap ever tries to hand it out again later - no double-allocation
path exists between the two engines.

### 7. Why running jobs are never touched

This phase is scoped to *routing new work*, never migrating or
interrupting existing work - `is_gpu_available` is the only gate a
GPU has to pass, and any assigned GPU fails it immediately regardless
of its utilization number. Nothing in `engine/balancing/` ever writes
to `GPU.assigned_user_id`, `GPU.assigned_job_id`, or `Job.status` for
a GPU that was already assigned before `route_job` was called -
`test_13_running_low_utilization_gpu_is_not_touched` asserts this
directly, and reclaiming a GPU because it looks idle stays entirely
Phase 4's decision, made independently and on its own sustained-
breach timeline.

### 8. Imbalance detection policy

Not `max(utilization) != min(utilization)` - the brief is explicit
that small differences are normal, not "imbalance". Imbalance is the
utilization spread *among available GPUs only*, compared against one
configured constant (`BalancingPolicy.imbalance_threshold_percent`,
default **30 percentage points** - `engine/balancing/config.py`).
Below the threshold, the router still always picks the least-utilized
available GPU (that's just "the obvious choice", not "balancing");
at or above it, the decision's `imbalance_detected` flag is set and
the reason string calls out the spread explicitly - this only affects
explainability/logging, never which GPU gets picked.

### 9. Tie-breaking

Documented once, used everywhere in the router: **utilization
percentage first, GPU id second** (lexicographic). Two GPUs at
exactly 20% never depend on dict ordering or insertion order -
`test_4_equal_utilization_breaks_tie_by_gpu_id` and
`test_4b_tie_break_is_stable_regardless_of_insertion_order` both pin
this down.

### 10. Complexity

| Operation | Complexity | Why |
|---|---|---|
| Identifying available GPUs | O(n) | One pass over the candidate pool (n = GPUs considered), calling `is_gpu_available` per GPU. |
| Utilization-safety filtering | O(a) | a = available GPUs from the step above; one `has_valid_utilization` check each. |
| Inserting candidates into the Min-Heap | O(r log r) total | r = routable GPUs; each `insert` is O(log r) (Phase 2's documented heap cost), done r times. |
| Selecting the least-utilized GPU | O(log r) | One `extract_min`. |
| HashMap lookup (selected id -> GPU) | O(1) average | Direct `HashMap.get`, same cost Phase 2 documents for it. |
| Imbalance spread calculation | O(a) | One pass over the available GPUs' utilization values. |
| Full `route_job` call | O(n + r log r) | Dominated by the O(n) filtering pass and the heap build - not asymptotically better than sorting for small r, but built on the project's actual Min-Heap rather than a bare `sorted()` call, and it's the same structure Phase 2 already built and documented for exactly this "find the minimum" job. |
| Committing via `finalize_assignment` | O(w) | w = jobs currently waiting - the same `_remove_from_waiting_queue` drain-and-rebuild `allocate_next` already pays (Phase 3, unchanged). |

### 11. Example routing trace

From `main.py`'s `demo_load_balancing()` - the brief's own critical
acceptance scenario: GPU-2 sits at 5% but is already Bob's, so GPU-4
(genuinely `IDLE`, 20%) is correctly preferred over it:

```
Pool: GPU-1=90% (assigned/A)  GPU-2=5% (assigned/B)  GPU-3=75% (assigned/C)  GPU-4=20% (IDLE/available)

Phase 3 (which job?): JOB-42 via FCFS - only one eligible candidate (JOB-42) - trivially FCFS
Phase 5 (which GPU?) candidates:
  GPU-1  util= 90.0%  assigned/unavailable
  GPU-2  util=  5.0%  assigned/unavailable
  GPU-3  util= 75.0%  assigned/unavailable
  GPU-4  util= 20.0%  available
Selected: GPU-4  Reason: GPU-4 has the lowest utilization (20.0%) among 1 available GPU(s)

Committed: JOB-42 -> GPU-4 (status=ACTIVE)
Bob's GPU-2 untouched: assigned_user=B  status=ACTIVE  utilization=5.0%
```

Event log for this decision: `BALANCE` (pool evaluated) ->
`BALANCE` (GPU-4 selected) -> `ALLOC` (assigned) -> `STATUS` (GPU-4
-> ACTIVE) - exactly the sequence the brief's documentation section
asks for.

## Scheduler orchestrator

`Scheduler` (`engine/scheduler.py`) is a small addition made *during*
Phase 6, not a new phase of policy: every phase's own architecture
diagram already draws one "Scheduler" box sitting above Allocation,
Reclamation and Balancing. Until Phase 6 that sequence only existed
informally, duplicated between `main.py`'s demo functions and a test
helper. `Scheduler` is that sequence written once - `submit_job`,
`try_allocate_all` (Phase 3 picks the job, Phase 5 picks the GPU,
commit), `record_utilization`/`respond_to_prompt`/
`check_reclamation_timeouts` (Phase 4), and one genuinely new
operation, `complete_job` (a job finishing normally releases its GPU
the same way a reclaim does, but logs a plain `STATUS` event instead
of `RECLAIM`, and never touches Phase 4's confirmation bookkeeping).
It adds no scheduling policy of its own - every decision still comes
from the engine that already owned it.

## Scenario Simulator (Phase 6)

### 1. Purpose

Demonstrating this project without real GPU hardware requires a
controlled way to feed the real engines a deterministic sequence of
inputs - GPUs, users, jobs arriving at chosen times, utilization
readings over chosen spans of (simulated) time, and user responses to
reclamation prompts - and then inspect exactly what the engines did
with them. `Simulator` is that harness. It will eventually power the
React frontend's demonstration mode; for now it is exercised entirely
from Python (`main.py`, and 57 tests in `tests/simulation/`).

### 2. Simulator vs. Scheduler vs. the three engines

|  | Decides | Never decides |
|---|---|---|
| **Simulator** | *When* to feed the scheduler its next input | Anything about who gets what |
| **Scheduler** | *The order* to call the three engines in | Any scheduling policy itself |
| **AllocationEngine** | *Which job* gets a GPU | *Which* GPU, or *when* to reclaim |
| **ReclamationEngine** | *When* a sustained-idle GPU is taken back | *Which* job replaces it |
| **LoadBalancingRouter** | *Which* available GPU a job goes to | *Which* job, or whether a GPU is reclaimable |

    The simulator generates inputs; the scheduler generates decisions.

Nothing in `engine/simulation/` contains a line like
`if scenario == "imbalance": selected_gpu = "GPU-4"`. A `Scenario`
only ever describes state and a timeline; `Simulator` only ever calls
`Scheduler` methods, which only ever call the three engines' real
decision methods.

### 3. The simulated clock (reusing, not duplicating, Phase 4's time model)

Every engine already accepted a plain `datetime` for "now" wherever
it reasoned about elapsed time - there was no separate clock class to
conflict with, only a convention (the caller decides what "now" is).
`SimulationClock` (`engine/simulation/clock.py`) is that convention's
source of truth during a simulation: `advance(timedelta)` moves
simulated time forward by an exact amount, never `time.sleep`s, so 25
simulated minutes or 3 simulated hours of sustained-low-utilization
history costs a fraction of a real second to produce.

**A real bug this exposed and fixed:** `AllocationEngine`'s commit
path (`_assign_gpu`) previously always called `datetime.now(timezone.utc)`
internally, ignoring whatever "now" its caller intended. That was
invisible while every caller was a human running a demo in real time,
but it broke the very first determinism test Phase 6 wrote (two runs
of the same scenario produced different `ALLOC`/`STATUS` timestamps).
`_assign_gpu`, `finalize_assignment`, `allocate_next`, and
`allocate_all` all now accept an optional `now` - still defaulting to
the real wall clock so every existing Phase 3 test is unaffected, but
`Scheduler.try_allocate_all` always passes its own (in a simulation,
simulated) `now` through explicitly.

### 4. Simulation speed

`SimulationClock.speed` is configuration, not behavior: `advance()`
never reads it. The backend simulator always advances by an exact,
caller-supplied `timedelta` - that is what "deterministic" requires.
`speed` exists for a *future* real-time-animated frontend to decide
how much simulated time one frame of actual wall-clock time should
correspond to (e.g. "1 real second = `speed` simulated minutes");
that mapping belongs to whatever eventually drives the clock from a
UI, not to the clock or the simulator itself, and nothing here
implements it yet.

### 5. Determinism

No randomness anywhere in `engine/simulation/`: no `random`, no
wall-clock reads feeding a decision, no unordered iteration a
decision depends on. `Scenario.fresh_copies()` deep-copies every GPU/
User/Job/action on every `reset()` (and on every `Simulator.__init__`)
so a previous run's in-place mutations can never leak into the next
one. `tests/simulation/test_determinism.py` runs every built-in
scenario twice (fresh `Simulator` each time, and via `reset()` on one
`Simulator`) and asserts every job's status/GPU, every GPU's state,
and the *entire* event log - including timestamps - are identical.

### 6. Scenario actions

Four action types (`engine/simulation/actions.py`), each a plain,
frozen dataclass carrying only an `offset` (simulated time after the
scenario's `start_time`) and the input to supply - never a decision:

| Action | Supplies | Calls |
|---|---|---|
| `AddJobAction` | a `Job` to submit | `Scheduler.submit_job` + `try_allocate_all` |
| `UtilizationAction` | one utilization reading for a GPU | `Scheduler.record_utilization` |
| `UserResponseAction` | YES/NO for a pending prompt | `Scheduler.respond_to_prompt` + `try_allocate_all` |
| `CompleteJobAction` | a job id that has finished | `Scheduler.complete_job` + `try_allocate_all` |

`Simulator._process_due_actions` fires every action whose absolute
due time (`start_time + offset`) is at or before the clock's current
time, in offset order, every time `advance()` is called - regardless
of how large or small the step was, so a coarse `run_to_completion`
step size never causes an action to be skipped or reordered relative
to another.

### 7. How utilization is supplied

Never randomly, and never computed by the simulator - a scenario
states the exact reading at an exact simulated time
(`UtilizationAction(offset=..., gpu_id=..., utilization_percent=...)`),
and `Simulator` hands it straight to `Scheduler.record_utilization`,
which updates the real `GPU.utilization_history` and feeds
`ReclamationEngine`'s per-GPU `UtilizationSlidingWindow` (Phase 2,
via Phase 4) exactly as if a real monitoring layer had reported it.
Whether that reading is "sustained" for long enough to matter is
entirely `is_sustained_breach`'s (Phase 4's) call - the simulator
never says "this GPU is idle".

### 8. How user responses are supplied

A `UserResponseAction(offset=..., gpu_id=..., response=ConfirmationResponse.YES|NO)`
calls `Scheduler.respond_to_prompt`, which raises if no prompt is
actually pending for that GPU - the simulator cannot answer a
question the Reclamation Engine never asked. "No response" needs no
action at all: it is simply the *absence* of a `UserResponseAction`,
combined with enough simulated time passing that
`Scheduler.check_reclamation_timeouts` (called on every `advance()`)
decides the grace period has elapsed - the real Phase 4 timeout path,
not a special simulator behavior.

### 9. The six scenarios

| id | Demonstrates |
|---|---|
| `gta5_excel` | Basic allocation: two users, two idle GPUs, arrivals two minutes apart. |
| `ml_video` | A nontrivial allocation-score decision: a small MEDIUM job beats a large HIGH job for the pool's only GPU. |
| `multiple_ml` | Queue pressure: five jobs (one `CRITICAL`), four GPUs - exactly one stays `WAITING`. |
| `idle_user` | Full Phase 4 flow: sustained Tier 1 breach -> prompt -> NO -> reclaim -> the reclaimed GPU serves an already-waiting job. |
| `imbalance` | Phase 5's core rule twice over: a 4%-but-assigned GPU is never touched; a later-freed GPU is compared against another available one by the Min-Heap. |
| `full_lifecycle` | The full 15-step demonstration below - allocation, routing, sustained reclamation, and reallocation, back to back. |

Five are the brief's required scenarios; `full_lifecycle` is the
brief's separately-requested "at least one end-to-end scenario".

### 10. End-to-end example (`full_lifecycle`)

```
Initial: GPU-1=90%/A  GPU-2=70%/B  GPU-3=4%/C  GPU-4=20%/IDLE

[09:05] REQUEST  J-D entered the scheduler
[09:05] BALANCE  Evaluated 4 GPU(s) in the pool for J-D - 1 available
[09:05] BALANCE  GPU-4 selected as lowest-utilization available GPU for J-D
[09:05] ALLOC    Assigned GPU-4 to User D
[09:05] STATUS   GPU-4 transitioned to ACTIVE
[09:32] REQUEST  J-E entered the scheduler
[09:32] BALANCE  No routable GPU for J-E                    <- nothing free yet
[09:35] PROMPT   GPU-3: are you still using this GPU?        <- Tier 1 sustained breach confirmed
[09:40] RESPONSE GPU-3: user responded NO
[09:40] RECLAIM  Reclaimed GPU-3 from User C
[09:40] BALANCE  GPU-3 selected as lowest-utilization available GPU for J-E
[09:40] ALLOC    Assigned GPU-3 to User E
[09:40] STATUS   GPU-3 transitioned to ACTIVE
```

GPU-3's 4% never mattered until Phase 4 independently confirmed it
was *sustained* and the user declined to keep it - exactly the
distinction Phase 5's README section draws between "underutilized"
and "available".

### 11. How this will eventually connect to React

`ScenarioRegistry.list_scenarios()` already returns exactly what a
scenario-picker UI needs (id, name, description) without React (or
anything else) hardcoding scenario names. `Simulator.snapshot()`
already returns the live `SchedulerState` a dashboard would render
(GPUs, users, jobs, assignments, event log) - deliberately not a
second, duplicate state shape. A future API layer only has to expose
`load`, `advance`/`tick`, `snapshot`, and the four action constructors
as endpoints; a speed slider maps to `SimulationClock.speed`
(descriptive today, load-bearing once a real-time-driven frontend
reads it). None of that transport layer exists yet - Phase 6 is
backend-only, as scoped.

### 12. Complexity

| Operation | Complexity | Why |
|---|---|---|
| `ScenarioRegistry.load` | O(1) | Dict lookup + one builder call. |
| `ScenarioRegistry.list_scenarios` | O(s) | s = registered scenarios; each is built once to read its name/description. |
| `Simulator.reset` | O(g + u + j + a log a) | g/u/j = initial GPUs/users/jobs (one `Scheduler` call each); a = actions, sorted once by offset. |
| `SimulationClock.advance` / `set` / `now` | O(1) | A single datetime addition/comparison. |
| `Simulator._process_due_actions` (per `advance`) | O(d) | d = actions due in this step; each executed exactly once. |
| One action's execution | Cost of the `Scheduler` call it makes | e.g. `AddJobAction` pays `try_allocate_all`'s O(m . (log n + k)) (Phase 3/5's own documented cost); `UtilizationAction` pays Phase 4's O(1) amortized + O(w) sustained-breach scan. |
| `Simulator.snapshot` | O(1) | Returns the live `SchedulerState` reference - never copied. |

## Backend API Adapter & React Dashboard (Phase 7)

### 1. Purpose

Phase 6 gave the project a deterministic simulator, but only Python
could drive it. Phase 7 puts a **terminal-styled control console** in
front of it: React visualizes exactly what `SchedulerState`/
`Simulator` already contain, and sends control commands (load a
scenario, start/pause/step, set speed, answer a reclamation prompt)
back to a small FastAPI adapter that calls the same `Scheduler`
Phase 6 already tested on its own.

### 2. Backend/frontend separation - the one rule that matters most

    No scheduling logic exists in React. React renders decisions; Python makes them.

Concretely, nothing in `frontend/src/` contains an allocation-score
formula, a utilization threshold, a GPU-selection rule, or a fake
event. Every number the dashboard shows - a GPU's status, a job's
priority, a waiting job's allocation score, an event's message - is
read verbatim from a backend response. The only "computation" any
component performs is display arithmetic that was already decided by
the backend (e.g. counting how many of the GPUs the backend labeled
`ACTIVE`) or pure formatting (seconds -> `"8m 12s"`). `api/serializers.py`
is the single place a backend dataclass becomes a JSON dict, and
`AllocationEngine.calculate_score` (Phase 3's real function) is what
computes the waiting-queue score - never duplicated in JavaScript.

```
Scheduler Core (engine/*)      <- untouched, framework-independent, still fully testable alone
       ^
Backend/API Adapter (api/*)    <- imports engine/; engine/ never imports this
       ^
HTTP (REST) + WebSocket
       ^
React (frontend/*)             <- renders state, sends commands, decides nothing
```

`engine/` was not moved into a `backend/` folder - it already existed
at the repository root as the framework-independent scheduler core
Phases 1-6 built and tested without any web dependency, and moving it
would only have renamed ~60 files without changing any behavior. `api/`
sits beside it and imports from it, which already satisfies "keep
core scheduler code separate from API/web code" - the separation is
by import direction, not by directory nesting.

### 3. Backend communication - REST for commands, WebSocket for the live stream

| | Used for |
|---|---|
| `GET /api/scenarios` | The scenario list, straight from `ScenarioRegistry.list_scenarios()`. |
| `GET /api/state` | The full dashboard snapshot (see `serialize_state`) - a plain poll-able fallback. |
| `POST /api/scenarios/{id}/load` | Load (and reset into) a scenario by id. 404 for an unknown id. |
| `POST /api/control/start` \| `/pause` \| `/reset` \| `/step` \| `/speed` | The simulation controls - each returns the resulting state. |
| `POST /api/prompt/respond` | `{gpu_id, response: "YES"|"NO"}` -> `Scheduler.respond_to_prompt` + `try_allocate_all`. 404 for an unknown GPU, 400 if that GPU has no prompt pending, 422 for an invalid response value. |
| `WS /ws/state` | Pushes the same `serialize_state` payload on connect, after every command, and once per background tick while the simulation is running - the one live stream every dashboard subscribes to. |

The core scheduler (`engine/`) never imports FastAPI/Starlette;
`api/session.py`'s `SimulationSession` is the only object that owns a
live `Simulator`, and every mutating method on it is a thin, validated
call straight into `Simulator`/`Scheduler` - the same pattern
`Scheduler` itself uses over Allocation/Reclamation/Balancing.

### 4. The simulated clock, live, over the network

`api/config.py` adds exactly one adapter-layer concept Phase 6 didn't
need on its own: a *background* loop (`asyncio.create_task`, one
iteration every `TICK_INTERVAL_SECONDS` = 0.5 real seconds) that calls
`SimulationSession.background_tick()`, which advances the simulated
clock by `speed` base ticks **only while the session is `running`** -
exactly mirroring `Simulator.advance`, just triggered by a repeating
timer instead of a human/test calling it. `speed` only ever multiplies
how much *simulated* time one background tick advances; it is read
nowhere near a scheduling decision (see Phase 6's own README section
on `SimulationClock.speed` for why that boundary already existed
before Phase 7).

### 5. Dashboard layout

Terminal/control-console aesthetic per the brief: near-black
background, monospace type, thin cyan borders, square corners, no
gradients or shadows.

```
┌─────────────────────────── HEADER ───────────────────────────┐
├──────────────┬──────────────────────────────┬────────────────┤
│ SCENARIO      │                              │                │
│ SELECTOR      │                              │                │
├──────────────┤        USERS PANEL           │  WAITING       │
│ SIMULATION    ├──────────────────────────────┤  QUEUE         │
│ CONTROLS      │                              │                │
├──────────────┤        GPU POOL              │                │
│ SYSTEM        │                              │                │
│ OVERVIEW      ├──────────────────────────────┤                │
├──────────────┤        EVENT LOG             │                │
│ ENGINE        │                              │                │
│ STATUS        │                              │                │
└──────────────┴──────────────────────────────┴────────────────┘
├─────────────────────────── FOOTER ────────────────────────────┤
```

(A `ReclamationPrompt` overlay appears on top of all of this whenever
`state.pending_prompts` is non-empty.)

### 6. GPU visualization

`GPUPool`/`GPURow`/`UtilizationBar`/`StatusBadge` render however many
GPUs the backend reports (never a hardcoded `GPU-1..GPU-4`), each
showing id, a utilization bar + percentage, assigned user/job, memory,
and a status badge colored by the project's fixed semantics:

| Status | Color |
|---|---|
| `ACTIVE` | green |
| `IDLE_WARNING` | yellow |
| `RECLAIMING` | red |
| `REALLOCATING` | blue |
| `IDLE` | neutral |

The bar's `width` has a CSS `transition` so two backend-reported
readings (e.g. 40% -> 55%) animate smoothly between them - this is
interpolation over values the backend sent, never a value React
invented in between.

### 7. Event log

`EventLog`/`EventEntry` render the backend's `events` list newest-
first, keyed by the backend's own `event_id` (already unique per
engine: `E...` from Allocation, `R...` from Reclamation, `B...` from
Balancing, `S...` from `Scheduler` - Phase 1-6 already assigned these,
so Phase 7 needed no new backend ID scheme). Because every WebSocket
`state` message carries the *entire* current `events` array rather
than an incremental delta, duplicate rendering is structurally
impossible - React's `key={event.event_id}` reconciliation just
matches existing DOM nodes instead of recreating them, and a dropped
message is self-healing on the very next broadcast.

### 8. Reclamation prompt

`ReclamationPrompt` renders only when the backend's `pending_prompts`
is non-empty (derived server-side from `ReclamationEngine.
has_pending_prompt` and the real `PROMPT` event already logged - never
frontend-guessed). Clicking YES/NO calls `POST /api/prompt/respond`
and renders whatever state comes back; there is no client-side timer
deciding a "no response" outcome - that remains
`ReclamationEngine`'s/the background loop's own grace-period check.

### 9. Waiting queue & allocation score

`WaitingQueue` renders `state.waiting_queue`, where each entry's
`allocation_score` was computed by `AllocationEngine.calculate_score`
(the real 0.6/0.4 formula) in `api/serializers.py` - never
`0.6 * priority + 0.4 * inverseSize` written in JavaScript. One
documented simplification: the displayed score is relative to *every*
currently-waiting job, whereas `select_next_job` internally narrows to
only critical jobs first when any are waiting - the number shown can
therefore differ slightly from the score that round's actual decision
used, but the decision itself is 100% engine-determined either way.

### 10. Why React contains no scheduling logic (and how that's enforced)

Every value on screen traces back to one of: a `serialize_*` function
in `api/serializers.py` (which only reads already-decided backend
fields, or calls back into real `engine.allocation`/`engine.
reclamation` code for the one derived number - allocation score), or
a pure formatter (`formatPercent`, `formatDuration`, ...) that performs
no scheduling arithmetic. `frontend/src/__tests__/api.test.js` locks
down the exact request shape every command sends (e.g. `respondToPrompt`
always POSTs `{gpu_id, response}` verbatim); nothing anywhere
constructs a `selectedGpu` or computes a priority-weighted score in JS.

### 11. State flow

```
Backend state (SchedulerState/Simulator)
    -> api/serializers.serialize_state          (adapter: dataclasses -> JSON)
    -> GET /api/state  or a WS "state" message
    -> useSchedulerState (the one React state hook)
    -> App.jsx passes state down as props
    -> components render only
```

```
User clicks a button (e.g. "NO" on a reclamation prompt)
    -> services/api.js respondToPrompt(gpuId, "NO")
    -> POST /api/prompt/respond
    -> SimulationSession.respond_to_prompt
    -> Scheduler.respond_to_prompt -> ReclamationEngine.respond   (the real decision)
    -> Scheduler.try_allocate_all                                  (lets a waiting job take the freed GPU)
    -> api/app.py returns + broadcasts the new serialize_state()
    -> useSchedulerState updates `state`
    -> every component re-renders from the new backend truth
```

### 12. Tests

**Backend** (`tests/api/test_app.py`, 18 tests, using FastAPI's
`TestClient` against the real app - no mocking of `Scheduler`/
`Simulator`): scenario list/load (incl. unknown id -> 404), the state
endpoint's full shape, GPU count changing with the loaded scenario
(never hardcoded), reset producing a different simulated time,
start/pause toggling, step advancing time, speed reaching the
simulator (and an invalid speed being rejected, not silently
clamped), a YES **and** a NO response each reaching the real
`ReclamationEngine` with the expected resulting GPU/job state, an
unknown-GPU or no-pending-prompt response being rejected safely, the
waiting queue's score coming from the real `AllocationEngine`, and
the WebSocket sending an initial state and then a fresh one after a
control command.

**Frontend** (`frontend/src/__tests__/`, 28 tests, vitest + Testing
Library) - deliberately data-flow tests, not "does JSX render":
`GPUPool.test.jsx` renders a *dynamic* number of GPUs from a fixture
payload and checks the exact utilization/assignment values and status
colors; `EventLog.test.jsx` checks backend event messages render
verbatim, newest-first, keyed by `event_id`; `ReclamationPrompt.test.jsx`
asserts clicking YES/NO calls the response handler with exactly
`(gpu_id, "YES"|"NO")` and nothing else; `api.test.js` mocks `fetch`
and asserts the exact method/path/JSON body each service function
sends, matching the FastAPI contract above; `WaitingQueue.test.jsx`,
`SimulationControls.test.jsx`, and `Footer.test.jsx` cover the
score/backend-driven-speed-range/disconnected-state requirements.

```
$ python -m pytest -v          # 307 passed (263 from Phases 1-6 + 18 Phase 7 + 26 Phase 8, all unchanged where prior)
$ cd frontend && npm test       # 28 passed
$ cd frontend && npm run build  # production build succeeds
```

## Hardware Abstraction (Phase 8, Part A)

> **Note on scope**: the Phase 8 instructions this section implements
> were cut off after Part A's architecture diagram - no Part B,
> testing list, or response format followed. This section covers what
> was fully specified (the monitoring abstraction) plus both goals
> the task explicitly stated up front (connect to real GPU signals
> through a clean interface; validate the pipeline end to end), rather
> than guessing at unstated further requirements.

### 1. Purpose

Every phase so far fed the scheduler through exactly one path:
`Scheduler.record_utilization`, called either directly or via a
scenario's `UtilizationAction` (Phase 6). Phase 8 adds two *real*
sources of that same call - `nvidia-smi` and NVML - behind one
interface, so the scheduler genuinely cannot tell (and never needs
code to tell) whether a reading came from a simulated scenario or
actual hardware:

```
Scheduler
    ^
GPUMonitor (interface, engine/hardware/monitor.py)
    ^
+--------------------+-------------------+------------------+
SimulatorGPUMonitor     NvidiaSMIMonitor     NVMLMonitor
(reads a Simulator's    (shells out to        (calls pynvml -
 current GPU state)      nvidia-smi)           direct kernel-driver API)
```

### 2. The interface

`GPUMonitor` (`engine/hardware/monitor.py`) has exactly one abstract
method, `get_gpu_metrics() -> List[GPUMetrics]`; `get_utilization`/
`get_memory`/`get_processes` are concrete conveniences built on top of
it, so every implementation gets them for free and can't disagree
with its own `get_gpu_metrics()` about a given GPU. `GPUMetrics`
(`engine/hardware/metrics.py`) is deliberately shaped like a richer
`UtilizationObservation` (Phase 1) - gpu_id, utilization, memory,
timestamp, processes - so nothing downstream needs a second shape.

### 3. The three implementations

- **`SimulatorGPUMonitor`** wraps a Phase 6 `Simulator` and reads its
  *current* GPU state - it generates no data of its own; a scenario's
  actions (or a test) already wrote the numbers onto the real `GPU`
  objects, this only reshapes them as `GPUMetrics`.
- **`NvidiaSMIMonitor`** shells out to `nvidia-smi --query-gpu=...
  --format=csv` and parses the output. Its `runner` (defaulting to
  `subprocess.run`) is injectable, so its real CSV-parsing logic is
  fully tested without `nvidia-smi` installed.
- **`NVMLMonitor`** calls `pynvml` (NVML - the direct API into the
  NVIDIA kernel driver, the same layer the project report describes
  as "the actual kernel level"). Its `pynvml_module` is injectable for
  the same reason.

Both real monitors raise `MonitorUnavailableError` - never a bare
exception, never a silent empty result - when their underlying
tool/library genuinely isn't present. `detect_gpu_monitor()`
(`engine/hardware/factory.py`) tries NVML then `nvidia-smi` and
returns the first one that both constructs *and* successfully
produces a reading (construction alone isn't a reliable check -
`NvidiaSMIMonitor` only discovers the binary is missing when it
actually tries to run it, which the factory's own test catches).

### 4. The bridge into the scheduler

`engine/hardware/poller.py`'s `feed_metrics(scheduler, metrics)` is
the *only* place a `GPUMetrics` becomes a
`Scheduler.record_utilization` call - the same entry point a
scenario's `UtilizationAction` already used. `MonitorPoller` wraps
that: pull metrics from any `GPUMonitor`, feed them in, then let the
scheduler react (`check_reclamation_timeouts`, `try_allocate_all`) -
the same observe -> decide -> update sequence every phase of this
project has used, now running on whichever monitor it was built with.

### 5. Validating the "source-agnostic" guarantee

`tests/hardware/test_source_agnostic.py` is the direct proof: two
independent `Scheduler`s, seeded identically, driven by (a) a
`SimulatorGPUMonitor` and (b) a monitor scripted with the exact same
numbers a real one would report - both reach the identical sustained-
breach -> `IDLE_WARNING` -> prompt -> `NO` -> reclaim outcome. The
scheduler's decision logic (Phases 3-5) was never touched to make
this true; it was already source-agnostic by construction (it only
ever consumed `datetime`/`float` values through `record_utilization`)
- Phase 8 only had to prove it, and to give real hardware a way in
that doesn't bypass that same call.

A genuine bug surfaced while building this: `detect_gpu_monitor`'s
first draft accepted `NvidiaSMIMonitor()` as "available" the moment it
*constructed*, but `NvidiaSMIMonitor.__init__` never touches
`nvidia-smi` at all - only `get_gpu_metrics()` does. On a machine with
no `nvidia-smi` installed (this one), the factory would have handed
back a monitor that fails on its very first real use. Fixed by having
the factory probe with one `get_gpu_metrics()` call before accepting
any candidate - `test_construction_only_failure_is_not_enough_to_
reject_a_candidate_falsely` locks this down.

### 6. Complexity

| Operation | Complexity | Why |
|---|---|---|
| `GPUMonitor.get_utilization`/`get_memory`/`get_processes` | O(g) | g = GPUs the monitor reports; a linear scan of `get_gpu_metrics()`'s result to find one id. |
| `SimulatorGPUMonitor.get_gpu_metrics` | O(g) | One pass building a `GPUMetrics` per GPU already in `SchedulerState`. |
| `NvidiaSMIMonitor.get_gpu_metrics` | O(g) | One subprocess call + one pass parsing g CSV lines. |
| `NVMLMonitor.get_gpu_metrics` | O(g) | One NVML call per GPU (count, handle, utilization, memory). |
| `detect_gpu_monitor` | O(c) | c = candidates tried (construct + one probe each) until one succeeds or all fail. |
| `feed_metrics` | O(m) | m = metrics in the batch; one dict lookup + one `record_utilization` call per known GPU. |
| `MonitorPoller.poll_once` | O(g) + Scheduler's own `try_allocate_all`/`check_reclamation_timeouts` cost | Dominated by whatever those two already cost (see their own phases' README sections), not by the monitor read itself. |

## Multi-User Demonstration System (Phase 9)

### 1. Login

`api/auth.py` seeds five demo accounts (`admin` role `ADMIN`;
`user_a`..`user_d` role `USER`) behind an opaque in-memory session
token - `POST /api/auth/login` (no password), `GET /api/auth/me`,
`POST /api/auth/logout`. **Every authenticated route resolves the
caller from this token, never from anything in a request body** -
`require_account`/`require_admin` (`api/app.py`) enforce this.

### 2. User Portal vs Admin Console - two views, one backend

`frontend/src/App.jsx` is the router: no session -> `Login`; logged
in as `ADMIN` -> `AdminConsole` (the pre-existing dashboard,
unchanged, plus a Decision Trace panel and a logout button); logged
in as `USER` -> the new `UserPortal` (request form, my jobs,
notifications, the same `ReclamationPrompt`). Both read from the
*same* `SimulationSession`/`Scheduler` - there is exactly one GPU
pool, one scheduler instance, regardless of how many browser tabs or
roles are connected.

### 3. GPU request -> real allocation, not a second scheduler

`SimulationSession.submit_user_request` (`api/session.py`) is the
literal missing piece a prior audit of this project flagged: nothing
previously let a live user reach `Scheduler.submit_job`. It now does,
via `POST /api/requests` - auto-registering the caller as a real
engine `User` on first use, then calling the exact same
`submit_job` + `try_allocate_all` sequence a scenario's own
`AddJobAction` already used. **Multi-GPU requests are rejected with a
clear 422, not faked** - `Job`/`AllocationEngine` only ever model one
GPU per job (a limitation already documented honestly in the Phase 8
audit).

### 4. Privacy is backend-enforced, not frontend-filtered

`serialize_portal_state` (`api/serializers.py`) builds a *different,
smaller* payload for a `USER` - only their own jobs, their own
pending prompt, their own notifications - never the full GPU pool,
other users, or the admin decision trace. The same enforcement
applies to `/ws/state`: a connection with a `USER` token receives the
scoped `portal_state` message type; no token (or an `ADMIN` token)
receives the full `state` message exactly as before Phase 9. A `USER`
answering `/api/prompt/respond` for a GPU they don't own gets a 403 -
checked via `SimulationSession.gpu_owner`, never trusted from the
request.

### 5. "Estimated completion" reuses the existing confirmation flow

Reaching a job's estimated duration does **not** mark it complete.
`ReclamationEngine.prompt_for_job_completion` + `Scheduler.
check_estimated_completions` raise the *same* `IDLE_WARNING` ->
`PROMPT` -> `respond` -> `RECLAIM`-or-back-off state machine the
utilization tiers already use (`tier` is simply `None` on a decision
raised this way, honestly reported as "not a utilization tier"
rather than faking one) - called from `SimulationSession.step`/
`background_tick` after every time advance.

### 6. Decision trace

`Scheduler.last_allocation_decision`/`last_routing_decision` (purely
observational fields, updated by `try_allocate_all`) back
`serialize_decision_trace`, shown in the Admin Console's new
`DecisionTrace` panel - the real candidates, scores, and reasons from
the most recent allocation, never recomputed for display.

### 7. Interactive Demo scenario

`interactive_demo` (`engine/simulation/scenarios.py`) is a blank
4-GPU pool with the four `USER` accounts already registered and *no*
scripted timeline - Mode 2 ("interactive users") alongside the five
pre-existing scripted scenarios (Mode 1).

### 8. Tests and results

50 new backend tests (`tests/test_scheduler.py`,
`tests/api/test_auth.py`, `tests/api/test_session.py`,
`tests/api/test_app_portal.py`, `tests/api/test_serializers_portal.py`,
plus scenario/determinism additions) and 11 new frontend tests
(`Login`, `GPURequestForm`, `MyJobs`, the auth-aware `api.js`
additions) - all passing, on top of every pre-existing test, which
still pass unchanged:

```
$ python -m pytest -q      # 361 passed
$ cd frontend && npm test    # 39 passed
```

The brief's own end-to-end demonstration (login -> User A requests a
GPU -> allocated -> Admin sees it -> User B requests -> waits ->
User A's job's estimated time elapses -> prompted -> NO -> reclaimed
-> User B's waiting job is allocated -> both views update -> full
event log) was run against the real, live FastAPI app and passes in
full - see the test suite and this phase's development history for
the exact trace.

## Multi-GPU Requests, the 10-GPU Pool & Resource Requests (Phase 10)

**Logical pool vs. physical hardware.** `interactive_demo` now models
the company's full **10-GPU logical pool** (`GPU-1`...`GPU-10`,
`engine/simulation/scenarios.py::build_interactive_demo`, all starting
FREE). This is a scheduler concept, not a claim that ten physical
NVIDIA cards exist on the machine running this project - see the next
section for how real hardware, when present, is a separate, optional
data *source* for however many of these logical GPUs it can see.

**Multi-GPU requests (1-9).** `Job` gained `gpu_count` (how many GPUs
a job needs) and `assigned_gpu_ids: List[str]` (how many it currently
holds), replacing the old single `assigned_gpu_id` field - kept as a
read-only, backward-compatible property (`assigned_gpu_ids[0]`) so
every pre-Phase-10 single-GPU call site kept working unchanged.
`AllocationEngine._assign_gpu` now *appends* to that list per GPU
committed and only moves a job to `RUNNING` (and off the waiting
queue) once `len(assigned_gpu_ids) >= gpu_count` - a partially-
satisfied multi-GPU job stays `WAITING` and keeps competing for its
*next* GPU under the exact same allocation policy (60/40 score, 20%
FCFS threshold, critical-job precedence) every single-GPU job always
has. `api/config.py` enforces `1 <= gpu_count <= 9` for a normal user
request (`MIN_USER_GPU_REQUEST`/`MAX_USER_GPU_REQUEST` -
`COMPANY_GPU_POOL_SIZE - 1`) - no user may request the entire pool.

**Partial availability -> resource requests.** When a multi-GPU
request cannot be fully satisfied from genuinely free GPUs,
`Scheduler._request_additional_gpus_if_needed` (called at the end of
every `try_allocate_all`) asks whoever holds one of the remaining
candidate GPUs to release it - through a *third* entry point
(`ReclamationEngine.request_gpu_for_reallocation`) into the exact same
prompt/respond/timeout confirmation state machine every utilization-
tier and estimated-completion prompt already uses, never a second
notification system. The holder gets the real choice (YES keeps it,
exactly like backing off a utilization prompt; NO releases it through
the same `_reclaim` a low-utilization breach uses) and a decline is
remembered per-requesting-job (`ReclamationEngine.has_declined_for`) so
the same job never immediately re-asks the same holder. This is scoped
to genuine multi-GPU requests only (`gpu_count > 1`) - an ordinary
single-GPU job simply waiting its turn never triggers it. Which job
actually receives a GPU freed this way is decided afterward by the
same one allocation policy, not by this method.

**Manual admin assignment.** `Scheduler.manual_assign_gpu` (exposed as
`POST /api/admin/assign`, admin-only) lets an admin place a user
directly onto a free GPU to establish a demo's starting arrangement -
implemented as a small placeholder `Job` committed through the same
`AllocationEngine.finalize_assignment` every real allocation uses, so
it is fully indistinguishable from one the engines assigned
themselves in every index and later decision. It is a one-time setup
action, not a second scheduling policy.

**Decision trace / event log.** `serialize_decision_trace` now also
reports the last resource-request decision (`scheduler.
last_resource_request_decision`); `serialize_gpu`/`serialize_pending_prompts`
expose who is asking (`requested_by_user_id/name`, `requested_for_job_id`)
for a resource-request prompt specifically, `None` for every ordinary
prompt. New `REQUEST`-type events narrate the ask itself
("User B needs 1 more GPU(s) for REQ-1; GPU-7 is currently held by
User D - requesting release") before the `PROMPT` event, so the full
sequence (request -> partial allocation -> candidate identified ->
prompt sent -> response -> reallocation) is genuinely readable from
the admin event log, not reconstructed after the fact.

**Real hardware wiring.** `SimulationSession.enable_real_hardware`
(exposed as `POST /api/hardware/enable`, admin-only) calls
`engine.hardware.factory.detect_gpu_monitor()` (NVML first, then
`nvidia-smi`) and, only if one genuinely initializes, switches
subsequent ticks onto it via the already-tested `MonitorPoller` -
identical entry point (`Scheduler.record_utilization`) a scenario's
own `UtilizationAction` uses, so the engines react the same way
regardless of source. `GET /api/hardware/status` always reports the
truth (`enabled`, `source`, `error`) rather than a client ever having
to guess. This development sandbox has no NVIDIA GPU, which the
session honestly reports (`tests/api/test_hardware_session.py`); the
same test suite proves the pipeline itself with an injected fake
monitor standing in for real NVML/`nvidia-smi` output.

The full acceptance scenario (10 free GPUs -> admin manually assigns
GPU-1/2->A, 3/4->B, 5/6->C, 7/8->D -> User C's portal shows exactly
GPU-5/6 -> User B requests 3 GPUs, gets 2 free ones (GPU-9/10) and is
short 1 -> the backend asks User D for GPU-7 -> User D sees the real
request in their own portal and releases it -> User B ends up RUNNING
with all 3 GPUs -> Admin Console agrees) was run against the real,
live FastAPI app and passes in full.

## What this project deliberately does NOT implement

- Any scheduling decision in React/JavaScript, anywhere.
- GPU migration or forcibly interrupting a running job to "improve"
  utilization.
- Actual GPU process termination/management, or `cgroups` enforcement.
- The pre-existing admin control endpoints (`/api/control/*`,
  `/api/scenarios/{id}/load`, `/api/state`) requiring login - they
  remain exactly as unauthenticated as before Phase 9, to avoid any
  regression to the 18 tests already covering them. Only the *new*
  Phase 9/10 endpoints (`/api/requests`, `/api/portal/state`,
  `/api/admin/assign`, `/api/hardware/*`) and `/api/prompt/respond`'s
  ownership check require/use a token. A fuller gate on the admin
  surface is a reasonable follow-up, not silently claimed here.
- Real hardware telemetry unless an admin explicitly enables it
  (`enable_real_hardware`) *and* `detect_gpu_monitor` genuinely finds
  NVML or `nvidia-smi` - never assumed, never faked.
- A second scheduler, a second notification system, or any
  scheduling-policy change for a multi-GPU/resource-request job: the
  60/40 allocation score, the 20% FCFS threshold, and the two-tier
  reclamation policy are exactly what they were before Phase 10.
- OAuth, a database, passwords, or anything beyond the explicitly
  simple demo login the brief asked for.
- Any workload-specific branching (`if job == "GTA5"` and similar).
- GPU leases of any kind.
- Ten physical NVIDIA cards - the 10-GPU pool is a logical scheduler
  concept; see the Phase 10 section above.
