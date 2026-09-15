# Dynamic GPU Resource Allocation Engine

**Team Vicimus** · Vishwakarma Institute of Technology (VIT), Pune
DSA Course Project

A backend-driven simulator and control system for allocating,
monitoring, and reclaiming GPUs from a shared, company-owned pool —
built to demonstrate that classical data structures and algorithms
(linked lists, heaps, priority queues, hash maps, queues, stacks,
sliding windows) are the actual load-bearing machinery behind a real
scheduling problem, not a syllabus checklist.

## Problem statement

Companies allocate GPUs from a shared pool, but *allocated* capacity
often doesn't match *utilized* capacity — a user holding two GPUs
might only be actively using one, while someone else waits for
capacity. There is no lease system and no fixed ownership: a GPU
should go to whoever needs it most right now, and idle capacity should
be detected and safely reclaimed back into the pool, without ever
yanking a GPU someone is genuinely still using.

## Project objective

Build a scheduling engine that:

1. **Allocates** available GPUs to waiting jobs using a documented,
   reproducible policy (a weighted priority/job-size score, with a
   size-similarity threshold that falls back to FCFS, and critical-job
   precedence).
2. **Monitors** utilization over time and **reclaims** GPUs that have
   been genuinely, sustainedly idle — never on a single low reading —
   through a confirm-before-reclaim flow, not an abrupt takeaway.
3. **Balances load** for new work across genuinely available GPUs,
   never touching a GPU that is already legitimately assigned no
   matter how idle it looks.
4. Does all of the above using hand-built, purpose-specific DSA
   structures, each justified by the exact decision it powers.
5. Is drivable from a live React dashboard, but computes **zero**
   scheduling logic in the frontend — every decision is made once, in
   Python, and only ever rendered on screen.

## High-level architecture

```
        React Frontend
              |
      REST  /  WebSocket
              |
   Python Scheduler Engine
              |
Allocation / Reclamation / Balancing
              |
    Hardware Monitoring Layer
              |
       NVML / nvidia-smi
              |
         NVIDIA GPU
```

The bottom of that chain is pluggable: a `GPUMonitor` interface has
three implementations — a deterministic **simulator** (the default,
used for all testing and demonstration), `nvidia-smi` (shells out to
the CLI tool), and NVML (`pynvml`, direct driver API). The scheduler
never knows or cares which one produced a reading — it only ever
consumes plain utilization values through one call
(`Scheduler.record_utilization`), so simulated and real hardware
produce identical decisions. This project runs entirely on the
simulator by default; no physical NVIDIA GPU is required or assumed.

Full details, diagrams, and worked traces for every layer are in
[`docs/architecture.md`](docs/architecture.md).

## Main components

| Component | Role |
|---|---|
| `engine/models/` | The domain model — `GPU`, `User`, `Job`, `GPUAssignment`, `Event`, `SchedulerState`. |
| `engine/dsa/` | Hand-built DSA structures (see below). |
| `engine/allocation/` | The Allocation Engine — decides *which job* gets a GPU. |
| `engine/reclamation/` | The Reclamation Engine — decides *when* a sustained-idle GPU is taken back. |
| `engine/balancing/` | The Load-Balancing Router — decides *which* available GPU a job goes to. |
| `engine/scheduler.py` | Thin orchestrator over the three engines above — no policy of its own. |
| `engine/simulation/` | Deterministic scenario simulator — drives the real engines through simulated time, without real hardware or real users. |
| `engine/hardware/` | The hardware abstraction layer — `GPUMonitor` + simulator/`nvidia-smi`/NVML implementations. |
| `api/` | FastAPI adapter — REST endpoints + the `/ws/state` WebSocket. Imports `engine/`; `engine/` never imports it. |
| `frontend/` | React + Vite dashboard — Admin Console and per-user Portal. Renders backend state and sends commands; computes no scheduling decisions. |
| `tests/`, `frontend/src/__tests__/` | Backend (pytest) and frontend (Vitest) test suites. |

## DSA concepts used

| Structure / concept | Used for |
|---|---|
| **Linked List** | The dynamic GPU pool (`GPUPool`) — O(1) insert/remove as GPUs join/leave. |
| **Min-Heap** | Selecting the least-utilized *available* GPU for load balancing. |
| **Max-Heap / Priority Queue** | Ordering waiting jobs by a supplied comparator (e.g. allocation score). |
| **HashMap** | O(1) user/job/GPU lookup (`UserGPUIndex` and friends). |
| **Queue** | FCFS ordering of waiting jobs. |
| **Stack** | Reclamation/release history, most-recent-first. |
| **Sliding Window** | Detecting *sustained* (not momentary) utilization breaches. |
| **Weighted scoring** | The 60/40 priority/job-size allocation-score formula driving allocation decisions. |
| **SJF (job-size-based scheduling)** | The size half of the allocation score, gated by a 20% job-size-similarity threshold. |
| **Load balancing** | Utilization-aware routing of new work across available GPUs. |

Full reasoning, complexity tables, and exactly where each structure is
wired into the engine: [`docs/dsa.md`](docs/dsa.md).

## Technology stack

- **Backend**: Python, FastAPI, Uvicorn, WebSockets, `pytest`.
- **Frontend**: React 18, Vite 5, Vitest 2 + Testing Library.
- **No database** — state lives in memory for the process lifetime
  (see [Known limitations](#known-limitations)).
- **No real NVIDIA hardware required** — optional NVML/`nvidia-smi`
  integration behind a common interface, simulator by default.

## Getting started

### Install dependencies

```bash
pip install -r requirements.txt        # backend
cd frontend && npm install             # frontend
```

### Run the backend

```bash
python main.py                                        # standalone demo trace, no server
python -m uvicorn api.app:app --reload --port 8000     # API server + WebSocket
```

### Run the frontend

```bash
cd frontend
npm run dev        # http://localhost:5173, talks to the backend on :8000
```

### Run the tests

```bash
python -m pytest -v          # backend — 427 tests
cd frontend && npm test        # frontend — component/service tests
```

Full setup notes (environment/config requirements, hardware
notes, dependency caveats): [`docs/setup.md`](docs/setup.md).

## Current project status

Phases 1 through 10 are complete and tested end to end:

1. Domain model → 2. DSA foundation → 3. Allocation Engine →
4. Reclamation Engine → 5. Load Balancing → 6. Scenario Simulator →
7. React dashboard + API adapter → 8. Hardware abstraction
(NVML/`nvidia-smi`/simulator) → 9. Multi-user demonstration system
(login, Admin Console, per-user Portal) → 10. Multi-GPU requests, the
10-GPU logical pool, and resource-request/release flows.

- **Backend**: 427 automated tests passing (`python -m pytest -v`).
- **Frontend**: full component/service test suite passing
  (`cd frontend && npm test`).
- A further 100-scenario acceptance/validation pass and a full
  written project report were also produced — see
  `gpu_scheduler_full_report.docx` and `final_report_dsa.docx` at the
  repository root.

See [`docs/architecture.md`](docs/architecture.md) for the full
phase-by-phase history, including specific bugs found and fixed along
the way.

## Known limitations

- **No persistence** — all state is in-memory for the process
  lifetime; restarting the server loses history. Not in scope for
  this course project.
- **No per-user GPU sub-pool / hard quota** — the pool is deliberately
  company-wide and shared; fairness comes from the allocation score
  and (documented) policy, not from partitioning.
- **The 20% job-size-similarity rule can let score outrank raw
  priority** for similarly-sized jobs — a deliberate, documented
  design choice, not a bug.
- **No physical NVIDIA hardware in this development environment** —
  the NVML/`nvidia-smi` paths are implemented and tested via dependency
  injection, but not validated against real hardware here.
- **Demo authentication is intentionally lightweight** (bearer token
  per username, no password storage) — sufficient for a course
  demonstration, explicitly not production-grade.
- **No leases of any kind, anywhere** — by design; see
  [`docs/architecture.md`](docs/architecture.md) for why.

Full list with rationale: the "What this project deliberately does
NOT implement" section in [`docs/architecture.md`](docs/architecture.md).

## Repository layout

```
engine/     Framework-independent scheduler core (models, DSA, allocation,
            reclamation, balancing, simulation, hardware abstraction)
api/        FastAPI adapter (REST + WebSocket) — imports engine/, never the reverse
frontend/   React + Vite dashboard (Admin Console + User Portal)
tests/      Backend test suite (pytest)
docs/       architecture.md, dsa.md, setup.md — see above
main.py     Standalone demo script (no server needed)
```
