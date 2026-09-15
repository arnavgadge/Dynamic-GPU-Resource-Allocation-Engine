from engine.balancing.config import BalancingPolicy
from engine.balancing.detector import is_pool_imbalanced, utilization_spread
from engine.models.gpu import GPU

POLICY = BalancingPolicy(imbalance_threshold_percent=30.0)


def make_gpu(gpu_id: str, utilization: float) -> GPU:
    return GPU(gpu_id=gpu_id, total_memory_mb=24_576, utilization_percent=utilization)


def test_spread_of_empty_or_single_gpu_is_zero():
    assert utilization_spread([]) == 0.0
    assert utilization_spread([make_gpu("GPU0", 50.0)]) == 0.0


def test_spread_is_max_minus_min():
    gpus = [make_gpu("GPU0", 20.0), make_gpu("GPU1", 65.0), make_gpu("GPU2", 5.0)]
    assert utilization_spread(gpus) == 60.0


def test_invalid_readings_are_excluded_from_spread():
    gpus = [make_gpu("GPU0", 20.0), make_gpu("GPU1", float("nan")), make_gpu("GPU2", 25.0)]
    assert utilization_spread(gpus) == 5.0


def test_small_difference_is_not_imbalanced():
    gpus = [make_gpu("GPU0", 20.0), make_gpu("GPU1", 25.0)]
    assert is_pool_imbalanced(gpus, POLICY) is False


def test_large_difference_is_imbalanced():
    gpus = [make_gpu("GPU0", 5.0), make_gpu("GPU1", 40.0)]
    assert is_pool_imbalanced(gpus, POLICY) is True


def test_exactly_at_threshold_counts_as_imbalanced():
    gpus = [make_gpu("GPU0", 10.0), make_gpu("GPU1", 40.0)]
    assert utilization_spread(gpus) == 30.0
    assert is_pool_imbalanced(gpus, POLICY) is True
