"""Test 12 from the Phase 10 brief: real GPU utilization flowing

    NVML/nvidia-smi -> hardware layer -> SchedulerState -> (WebSocket/API) -> React

exercised at the `SimulationSession` level, the same object every
route in `api/app.py` drives. This dev machine has no NVIDIA hardware,
so `enable_real_hardware` is also verified to honestly report that
rather than pretending - and the *pipeline itself* is proven with an
injected fake monitor standing in for real NVML/nvidia-smi output,
exactly the substitution `engine/hardware`'s own tests already use.
"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.app import app
from api.session import SimulationSession
from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError
from engine.models.enums import GPUStatus
from engine.simulation import load_default_registry


class _FakeRealMonitor(GPUMonitor):
    """Stands in for `NVMLMonitor`/`NvidiaSMIMonitor` - same interface,
    same `GPUMetrics` shape, just no real driver/library underneath.
    """

    def __init__(self, readings):
        self._readings = readings

    def get_gpu_metrics(self):
        return self._readings


def test_this_dev_machine_honestly_reports_no_real_gpu_hardware():
    session = SimulationSession(load_default_registry(), "interactive_demo")
    available = session.enable_real_hardware()

    # This is a course-project dev sandbox, not an NVIDIA workstation -
    # the real assertion is that the session tells the truth about it
    # rather than fabricating a reading.
    assert available is False
    assert session.hardware_monitor is None
    assert session.hardware_error is not None and session.hardware_error != ""
    assert session.hardware_source is None


def test_injected_monitor_reading_flows_into_scheduler_state():
    """Proves the pipeline itself: a `GPUMonitor` -> `Scheduler.
    record_utilization` -> `GPU.utilization_percent` -> (from here,
    already-tested serializers/WebSocket carry it to React unchanged).
    """
    session = SimulationSession(load_default_registry(), "interactive_demo")
    job = session.submit_user_request("user_a", "User A", "job", 1, 500, "MEDIUM")
    gpu_id = job.assigned_gpu_id

    session.hardware_monitor = _FakeRealMonitor([
        GPUMetrics(
            gpu_id=gpu_id, utilization_percent=73.5, memory_used_mb=4096.0,
            memory_total_mb=24_576.0, timestamp=session.simulator.clock.now(),
        ),
    ])
    session.hardware_source = "_FakeRealMonitor"
    # Phase 16 hardening made the poller persistent (its `health`
    # accumulates across ticks) rather than rebuilt every call -
    # injecting the fake monitor directly (bypassing enable_real_
    # hardware, which real code always goes through) means this test
    # must also wire up the poller itself.
    from engine.hardware.poller import MonitorPoller
    session._hardware_poller = MonitorPoller(session.simulator.scheduler, session.hardware_monitor)

    session.step()

    gpu = session.simulator.scheduler.state.get_gpu(gpu_id)
    assert gpu.utilization_percent == 73.5
    assert gpu.memory_used_mb == 4096.0
    assert gpu.status == GPUStatus.ACTIVE  # 73.5% is nowhere near either reclamation tier


def test_hardware_status_and_admin_only_enable_endpoint():
    with TestClient(app) as client:
        status = client.get("/api/hardware/status").json()
        assert status["enabled"] is False

        login = client.post("/api/auth/login", json={"username": "user_a"}).json()
        forbidden = client.post(
            "/api/hardware/enable", headers={"Authorization": f"Bearer {login['token']}"},
        )
        assert forbidden.status_code == 403

        admin = client.post("/api/auth/login", json={"username": "admin"}).json()
        response = client.post(
            "/api/hardware/enable", headers={"Authorization": f"Bearer {admin['token']}"},
        )
        assert response.status_code == 200
        body = response.json()
        # Honest either way: no real GPU here, so this must report
        # disabled with a reason, never a fabricated "enabled": true.
        assert body["enabled"] is False
        assert body["error"]
