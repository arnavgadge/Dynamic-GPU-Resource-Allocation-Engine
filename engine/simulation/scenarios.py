"""The project's built-in demo scenarios.

Every scenario here only *describes state and a timeline of inputs* -
GPUs, users, jobs, utilization readings, user responses - at specific
simulated-time offsets. None of them contain a scheduling decision:
which job wins, which GPU is chosen, and when a GPU is reclaimed are
always left for the real engines (via `Simulator`/`Scheduler`) to
decide once the scenario runs. A workload label like ``"GTA5"`` or
``"ML Training"`` is metadata on a `Job.name` for display only -
nothing here, or anywhere the engines read a `Job`, branches on it.
"""

from datetime import datetime, timedelta, timezone

from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.simulation.actions import AddJobAction, CompleteJobAction, UserResponseAction, UtilizationAction
from engine.simulation.registry import ScenarioRegistry
from engine.simulation.scenario import Scenario

#: Shared start time for every built-in scenario - arbitrary but fixed,
#: so nothing about a scenario depends on when it happens to be run.
_START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def _at(minutes: float) -> datetime:
    return _START + timedelta(minutes=minutes)


# ------------------------------------------------------------------
# Scenario 1 - GTA5 + Excel: basic allocation
# ------------------------------------------------------------------

def build_gta5_excel() -> Scenario:
    gpus = [
        GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.IDLE),
        GPU(gpu_id="GPU-2", total_memory_mb=24_576, status=GPUStatus.IDLE),
    ]
    users = [
        User(user_id="alice", name="Alice", priority=Priority.MEDIUM),
        User(user_id="bob", name="Bob", priority=Priority.MEDIUM),
    ]
    actions = [
        AddJobAction(
            offset=timedelta(0),
            job=Job(job_id="J-GTA5", user_id="alice", name="GTA5", priority=Priority.MEDIUM,
                     estimated_size_minutes=30, submitted_at=_at(0)),
        ),
        AddJobAction(
            offset=timedelta(minutes=2),
            job=Job(job_id="J-EXCEL", user_id="bob", name="Excel", priority=Priority.MEDIUM,
                     estimated_size_minutes=25, submitted_at=_at(2)),
        ),
    ]
    return Scenario(
        scenario_id="gta5_excel",
        name="GTA5 + Excel",
        description=(
            "Two users submit ordinary jobs (labelled only for the demo) a couple of "
            "minutes apart onto a pool of two idle GPUs - the simplest possible "
            "allocation walkthrough: waiting queue -> Allocation Engine -> available "
            "GPU -> assignment -> events."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
        actions=actions,
    )


# ------------------------------------------------------------------
# Scenario 2 - ML Training + Video Editing: a nontrivial score
# ------------------------------------------------------------------

def build_ml_video() -> Scenario:
    gpus = [GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
                 assigned_user_id="seed", assigned_job_id="J-SEED")]
    users = [
        User(user_id="seed", name="Warmup", priority=Priority.LOW),
        User(user_id="dave", name="Dave", priority=Priority.HIGH),
        User(user_id="erin", name="Erin", priority=Priority.MEDIUM),
    ]
    # A placeholder job already occupying the pool's only GPU, so the
    # two real candidates below are forced to actually wait - and
    # compete - before a GPU is even available.
    initial_jobs = [
        Job(job_id="J-SEED", user_id="seed", name="Warmup", priority=Priority.LOW,
             estimated_size_minutes=999, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-1"]),
    ]
    actions = [
        AddJobAction(
            offset=timedelta(0),
            job=Job(job_id="J-ML", user_id="dave", name="ML Training", priority=Priority.HIGH,
                     estimated_size_minutes=180, submitted_at=_at(0)),
        ),
        AddJobAction(
            offset=timedelta(minutes=2),
            job=Job(job_id="J-VIDEO", user_id="erin", name="Video Editing", priority=Priority.MEDIUM,
                     estimated_size_minutes=20, submitted_at=_at(2)),
        ),
        # Frees the only GPU once both candidates are already waiting,
        # so the Allocation Engine has to choose between them.
        CompleteJobAction(offset=timedelta(minutes=5), job_id="J-SEED"),
    ]
    return Scenario(
        scenario_id="ml_video",
        name="ML Training + Video Editing",
        description=(
            "A HIGH-priority, large ML Training job and a MEDIUM-priority, small "
            "Video Editing job both end up waiting for the pool's only GPU at the "
            "same time - a deliberately nontrivial case for the allocation-score "
            "formula (0.6 x priority + 0.4 x size-inverse), not pure priority or pure SJF."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
        initial_jobs=initial_jobs,
        actions=actions,
    )


# ------------------------------------------------------------------
# Scenario 3 - multiple ML jobs: waiting-queue pressure
# ------------------------------------------------------------------

def build_multiple_ml() -> Scenario:
    gpus = [GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE) for i in range(1, 5)]
    users = [User(user_id=f"user-{i}", name=f"User {i}", priority=priority) for i, priority in enumerate(
        [Priority.HIGH, Priority.MEDIUM, Priority.MEDIUM, Priority.LOW, Priority.CRITICAL], start=1
    )]
    initial_jobs = [
        Job(job_id="J-1", user_id="user-1", name="ML Training A", priority=Priority.HIGH,
             estimated_size_minutes=45, submitted_at=_at(0)),
        Job(job_id="J-2", user_id="user-2", name="ML Training B", priority=Priority.MEDIUM,
             estimated_size_minutes=40, submitted_at=_at(0)),
        Job(job_id="J-3", user_id="user-3", name="ML Training C", priority=Priority.MEDIUM,
             estimated_size_minutes=200, submitted_at=_at(0)),
        Job(job_id="J-4", user_id="user-4", name="ML Training D", priority=Priority.LOW,
             estimated_size_minutes=15, submitted_at=_at(0)),
        Job(job_id="J-5", user_id="user-5", name="ML Training E (critical)", priority=Priority.CRITICAL,
             estimated_size_minutes=60, submitted_at=_at(0)),
    ]
    return Scenario(
        scenario_id="multiple_ml",
        name="Multiple ML Jobs",
        description=(
            "Five users submit ML training jobs at once onto a pool of only four "
            "GPUs - one job is guaranteed to stay WAITING. Exercises the Priority "
            "Queue, critical-job precedence, the allocation score, and GPU routing "
            "across a full round of allocation, without hardcoding who loses."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
        initial_jobs=initial_jobs,
    )


# ------------------------------------------------------------------
# Scenario 4 - idle user: Phase 4 reclamation end to end
# ------------------------------------------------------------------

def build_idle_user() -> Scenario:
    gpus = [GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
                 assigned_user_id="frank", assigned_job_id="J-LONGTAIL")]
    users = [
        User(user_id="frank", name="Frank", priority=Priority.MEDIUM),
        User(user_id="grace", name="Grace", priority=Priority.MEDIUM),
    ]
    initial_jobs = [
        Job(job_id="J-LONGTAIL", user_id="frank", name="Long-running job", priority=Priority.MEDIUM,
             estimated_size_minutes=600, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-1"]),
    ]
    # Six readings, five minutes apart, span exactly the default Tier 1
    # window (25 minutes) - see engine/reclamation/policy.py.
    actions = [
        UtilizationAction(offset=timedelta(minutes=m), gpu_id="GPU-1", utilization_percent=1.0)
        for m in (0, 5, 10, 15, 20, 25)
    ] + [
        # Grace arrives early and simply has to wait - GPU-1 is the
        # only GPU in this scenario, and it's still legitimately busy.
        AddJobAction(
            offset=timedelta(minutes=2),
            job=Job(job_id="J-NEWWORK", user_id="grace", name="New work", priority=Priority.MEDIUM,
                     estimated_size_minutes=20, submitted_at=_at(2)),
        ),
        # Frank explicitly says "no" once prompted (the prompt fires
        # the moment the sustained breach is confirmed, at minute 25).
        UserResponseAction(offset=timedelta(minutes=30), gpu_id="GPU-1", response=ConfirmationResponse.NO),
    ]
    return Scenario(
        scenario_id="idle_user",
        name="Idle User",
        description=(
            "Frank's GPU sits below 2% utilization for the full Tier 1 sustained "
            "window - the Sliding Window confirms it, the Reclamation Engine prompts "
            "him, and he answers NO. The GPU is reclaimed and Grace's already-waiting "
            "job receives it - entirely through Phase 4 and Phase 3/5, never a direct "
            "state edit."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
        initial_jobs=initial_jobs,
        actions=actions,
    )


# ------------------------------------------------------------------
# Scenario 5 - imbalance: Phase 5 routing
# ------------------------------------------------------------------

def build_imbalance() -> Scenario:
    gpus = [
        GPU(gpu_id="GPU-1", total_memory_mb=24_576, utilization_percent=90.0, status=GPUStatus.ACTIVE,
             assigned_user_id="user-a", assigned_job_id="J-A"),
        GPU(gpu_id="GPU-2", total_memory_mb=24_576, utilization_percent=80.0, status=GPUStatus.ACTIVE,
             assigned_user_id="user-b", assigned_job_id="J-B"),
        GPU(gpu_id="GPU-3", total_memory_mb=24_576, utilization_percent=4.0, status=GPUStatus.ACTIVE,
             assigned_user_id="user-c", assigned_job_id="J-C"),
        GPU(gpu_id="GPU-4", total_memory_mb=24_576, utilization_percent=50.0, status=GPUStatus.IDLE),
    ]
    users = [User(user_id=uid, name=name, priority=Priority.MEDIUM) for uid, name in
              [("user-a", "User A"), ("user-b", "User B"), ("user-c", "User C"), ("user-d", "User D")]]
    initial_jobs = [
        Job(job_id="J-A", user_id="user-a", name="existing", priority=Priority.MEDIUM,
             estimated_size_minutes=120, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-1"]),
        Job(job_id="J-B", user_id="user-b", name="existing", priority=Priority.MEDIUM,
             estimated_size_minutes=120, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-2"]),
        Job(job_id="J-C", user_id="user-c", name="existing", priority=Priority.MEDIUM,
             estimated_size_minutes=120, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-3"]),
    ]
    actions = [
        # Stage 1: only GPU-4 is genuinely available - GPU-3's 4% must
        # not matter at all.
        AddJobAction(
            offset=timedelta(minutes=1),
            job=Job(job_id="J-D", user_id="user-d", name="new work", priority=Priority.MEDIUM,
                     estimated_size_minutes=15, submitted_at=_at(1)),
        ),
        # Stage 2: GPU-2 finishes and is re-reported at a fresh, low
        # reading - now two GPUs are genuinely available with
        # different utilization, so the Min-Heap has an actual choice.
        CompleteJobAction(offset=timedelta(minutes=10), job_id="J-B"),
        UtilizationAction(offset=timedelta(minutes=11), gpu_id="GPU-2", utilization_percent=10.0),
        AddJobAction(
            offset=timedelta(minutes=12),
            job=Job(job_id="J-E", user_id="user-d", name="more new work", priority=Priority.MEDIUM,
                     estimated_size_minutes=15, submitted_at=_at(12)),
        ),
    ]
    return Scenario(
        scenario_id="imbalance",
        name="Imbalance",
        description=(
            "GPU-3 sits at only 4% but is legitimately assigned to User C - a new "
            "job must go to GPU-4 (the only genuinely available GPU), never GPU-3. "
            "A second stage then frees GPU-2 at a low utilization alongside GPU-4, "
            "demonstrating the Min-Heap choosing between two genuinely available GPUs."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
        initial_jobs=initial_jobs,
        actions=actions,
    )


# ------------------------------------------------------------------
# Scenario 6 - full lifecycle: the end-to-end demonstration
# ------------------------------------------------------------------

def build_full_lifecycle() -> Scenario:
    gpus = [
        GPU(gpu_id="GPU-1", total_memory_mb=24_576, utilization_percent=90.0, status=GPUStatus.ACTIVE,
             assigned_user_id="user-a", assigned_job_id="J-A"),
        GPU(gpu_id="GPU-2", total_memory_mb=24_576, utilization_percent=70.0, status=GPUStatus.ACTIVE,
             assigned_user_id="user-b", assigned_job_id="J-B"),
        GPU(gpu_id="GPU-3", total_memory_mb=24_576, utilization_percent=4.0, status=GPUStatus.ACTIVE,
             assigned_user_id="user-c", assigned_job_id="J-C"),
        GPU(gpu_id="GPU-4", total_memory_mb=24_576, utilization_percent=20.0, status=GPUStatus.IDLE),
    ]
    users = [User(user_id=uid, name=name, priority=Priority.MEDIUM) for uid, name in
              [("user-a", "User A"), ("user-b", "User B"), ("user-c", "User C"),
               ("user-d", "User D"), ("user-e", "User E")]]
    initial_jobs = [
        Job(job_id="J-A", user_id="user-a", name="existing", priority=Priority.MEDIUM,
             estimated_size_minutes=300, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-1"]),
        Job(job_id="J-B", user_id="user-b", name="existing", priority=Priority.MEDIUM,
             estimated_size_minutes=300, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-2"]),
        Job(job_id="J-C", user_id="user-c", name="existing", priority=Priority.MEDIUM,
             estimated_size_minutes=300, status=JobStatus.RUNNING, started_at=_at(0),
             submitted_at=_at(0), assigned_gpu_ids=["GPU-3"]),
    ]
    actions = [
        # Step 6-7: a new job arrives; only GPU-4 is genuinely
        # available, so the Load Balancer must choose it over GPU-3
        # despite GPU-3's much lower utilization.
        AddJobAction(
            offset=timedelta(minutes=5),
            job=Job(job_id="J-D", user_id="user-d", name="new work", priority=Priority.HIGH,
                     estimated_size_minutes=30, submitted_at=_at(5)),
        ),
        # Step 8-9: GPU-3 sustains below 2% for the full Tier 1 window.
        UtilizationAction(offset=timedelta(minutes=10), gpu_id="GPU-3", utilization_percent=1.0),
        UtilizationAction(offset=timedelta(minutes=15), gpu_id="GPU-3", utilization_percent=1.0),
        UtilizationAction(offset=timedelta(minutes=20), gpu_id="GPU-3", utilization_percent=1.0),
        UtilizationAction(offset=timedelta(minutes=25), gpu_id="GPU-3", utilization_percent=1.0),
        UtilizationAction(offset=timedelta(minutes=30), gpu_id="GPU-3", utilization_percent=1.0),
        UtilizationAction(offset=timedelta(minutes=35), gpu_id="GPU-3", utilization_percent=1.0),
        # Step: another job arrives while nothing is free yet.
        AddJobAction(
            offset=timedelta(minutes=32),
            job=Job(job_id="J-E", user_id="user-e", name="waiting work", priority=Priority.MEDIUM,
                     estimated_size_minutes=40, submitted_at=_at(32)),
        ),
        # Steps 10-14: prompted at minute 35 (sustained breach
        # confirmed); User C says no; GPU-3 is reclaimed, goes IDLE,
        # and J-E (the only waiting job) receives it automatically.
        UserResponseAction(offset=timedelta(minutes=40), gpu_id="GPU-3", response=ConfirmationResponse.NO),
    ]
    return Scenario(
        scenario_id="full_lifecycle",
        name="Full Lifecycle",
        description=(
            "The complete demonstration: an initial pool with mixed assignment and "
            "utilization, a new job routed to the only genuinely available GPU "
            "(never the merely-underutilized one), a sustained Tier 1 breach on an "
            "assigned GPU, a declined confirmation prompt, reclamation back to IDLE, "
            "and a waiting job immediately receiving the newly-freed GPU - all "
            "through the real Allocation/Reclamation/Balancing engines."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
        initial_jobs=initial_jobs,
        actions=actions,
    )


def build_interactive_demo() -> Scenario:
    """A blank-canvas pool for the interactive demonstration mode.

    Ten GPUs (the company's full logical pool - Phase 10, Requirement
    2), no pre-loaded jobs, no scripted actions - unlike every
    scenario above, this one describes *no* timeline at all. It exists
    so a logged-in demo user ("real demo users... submit requests...
    the backend processes them normally") has a clean, predictable
    pool to request against, instead of walking into one of the other
    scenarios' already-scripted story. The four users match the login
    accounts (`api/auth.py`) so a request submitted under a logged-in
    identity lands on a `User` the engine already knows, with a real
    priority - not an auto-provisioned placeholder.

    All ten GPUs start FREE (`GPUStatus.IDLE`, unassigned) - the admin
    then manually establishes whatever initial arrangement a
    demonstration needs (`SimulationSession.manual_assign_gpu`) rather
    than one being hardcoded here.
    """
    gpus = [GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE) for i in range(1, 11)]
    users = [
        User(user_id="user_a", name="User A", priority=Priority.HIGH),
        User(user_id="user_b", name="User B", priority=Priority.MEDIUM),
        User(user_id="user_c", name="User C", priority=Priority.MEDIUM),
        User(user_id="user_d", name="User D", priority=Priority.LOW),
    ]
    return Scenario(
        scenario_id="interactive_demo",
        name="Interactive Demo",
        description=(
            "The company's full 10-GPU logical pool, all free, with no pre-loaded "
            "jobs - for an admin to manually establish a starting arrangement and "
            "logged-in demo users (User A-D) to submit real (including multi-GPU) "
            "requests through the User Portal, watching the actual Allocation/"
            "Reclamation/Balancing engines respond instead of a pre-scripted timeline."
        ),
        start_time=_START,
        gpus=gpus,
        users=users,
    )


def register_builtin_scenarios(registry: ScenarioRegistry) -> None:
    """Register every scenario above into ``registry`` by id."""
    registry.register("gta5_excel", build_gta5_excel)
    registry.register("ml_video", build_ml_video)
    registry.register("multiple_ml", build_multiple_ml)
    registry.register("idle_user", build_idle_user)
    registry.register("imbalance", build_imbalance)
    registry.register("full_lifecycle", build_full_lifecycle)
    registry.register("interactive_demo", build_interactive_demo)
