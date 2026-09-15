# Setup & Running

## Requirements

- **Python** 3.10+ (developed/tested against 3.14).
- **Node.js** 18+ and npm (for the frontend — Vite 5 / Vitest 2).
- No NVIDIA hardware is required. Real-hardware monitoring (NVML /
  `nvidia-smi`) is optional and fully guarded — see
  [Hardware notes](#hardware-notes) below.

## 1. Backend — install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, HTTPX, `websockets`, and `pytest` —
everything the engine, API and test suite need. `pynvml` is
deliberately **not** pinned as a hard dependency: it is only imported
inside `engine/hardware/nvml_monitor.py`, guarded so its absence never
breaks anything else, and is only needed if you want to exercise real
NVML-based GPU monitoring on a machine that actually has an NVIDIA
GPU + driver + `pynvml` installed.

## 2. Backend — run the demo script

```bash
python main.py
```

Runs the sample state through the DSA structures, then the
Allocation, Reclamation, and Load-Balancing engines in turn, then the
`full_lifecycle` scenario end-to-end through the deterministic
simulator — printing a full trace of each decision to the terminal.
No GPU hardware or server needed.

## 3. Backend — run the API server

```bash
python -m uvicorn api.app:app --reload --port 8000
```

Starts the FastAPI app (REST endpoints + the `/ws/state` WebSocket)
on `http://localhost:8000`, backed by the real `Scheduler`/`Simulator`
— nothing mocked.

## 4. Frontend — install & run

```bash
cd frontend
npm install
npm run dev
```

Opens the dashboard on `http://localhost:5173`, talking to the
backend on `:8000`. The scenario selector defaults to `gta5_excel`;
pick another scenario (e.g. `idle_user`, `full_lifecycle`) and press
**START** to watch a full allocation → reclamation → reallocation
lifecycle happen live.

`npm run build` produces a production build in `frontend/dist/`
(already gitignored).

## 5. Running the tests

**Backend** (from the repository root):

```bash
python -m pytest -v
```

Runs every unit + integration test under `tests/` — 427 tests as of
this writing (`engine/`, `api/`, and their submodules), configured via
[`pytest.ini`](../pytest.ini) (`pythonpath = .`, `testpaths = tests`).

**Frontend** (from `frontend/`):

```bash
npm test
```

Runs the Vitest + Testing Library suite (component + service tests
under `frontend/src/__tests__/`).

## Environment / configuration notes

- No `.env` file or secret configuration is required to run the
  project as-is — the demo login (`api/auth.py`) uses fixed, in-memory
  demo accounts (`admin`, `user_a`..`user_d`), and every other setting
  (tick interval, allowed simulation speeds, GPU pool size, reclamation
  thresholds) is a plain Python constant in `api/config.py` /
  `engine/*/config.py`, not an environment variable.
- `frontend/vite.config.js` hardcodes dev server port `5173`; the
  frontend's `services/api.js` expects the backend on `:8000` — if you
  need different ports, those are the two places to change (not
  covered by an env file today).

## Hardware notes

Real GPU monitoring (`POST /api/hardware/enable`, admin-only) tries
NVML (`pynvml`) first, then falls back to shelling out to
`nvidia-smi`, and — if neither is available, as on this development
machine — honestly reports `enabled: false` with a reason, rather
than fabricating a reading. The scheduler's own decision logic never
depends on which source (simulator, NVML, or `nvidia-smi`) produced a
utilization reading; see `tests/hardware/test_source_agnostic.py`.

## Known dependency/configuration gaps

- There was previously no `requirements.txt` documenting the backend's
  runtime dependencies (FastAPI/Uvicorn/HTTPX/websockets/pytest) —
  the project ran only because those packages happened to already be
  installed in the developer's environment. `requirements.txt` at the
  repository root now fixes this.
- There is no `pyproject.toml`/`setup.py` — the project is run
  in-place (`pythonpath = .` in `pytest.ini`, `python main.py` from
  the repo root), not installed as a package. That's consistent with
  its current scope as a course project and was left unchanged.
