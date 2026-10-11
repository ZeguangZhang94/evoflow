"""迭代熔断测试：超轮熔断、连续回退熔断、正常演进不熔断。"""

import pytest

from evoflow.core.discipline import IterationController, IterationHalted


def test_halt_when_exceeding_max_rounds():
    c = IterationController(max_rounds=3)
    c.record_round(0.8)
    c.record_round(0.85)
    c.record_round(0.9)
    with pytest.raises(IterationHalted, match="max_rounds"):
        c.record_round(0.95)


def test_default_max_rounds_is_3():
    assert IterationController().max_rounds == 3


def test_halt_on_consecutive_regressions():
    c = IterationController(max_rounds=10, max_consecutive_regressions=2)
    c.record_round(0.9)
    c.record_round(0.8)  # 第 1 次回退
    with pytest.raises(IterationHalted, match="连续回退"):
        c.record_round(0.7)  # 第 2 次连续回退 -> 熔断


def test_single_regression_does_not_halt():
    c = IterationController(max_rounds=5)
    c.record_round(0.9)
    c.record_round(0.8)  # 单次回退允许
    c.record_round(0.85)  # 恢复上升，回退计数清零
    c.record_round(0.84)  # 再次单次回退
    assert c.rounds == 4


def test_invalid_config():
    with pytest.raises(ValueError):
        IterationController(max_rounds=0)
    with pytest.raises(ValueError):
        IterationController(max_rounds=3, max_consecutive_regressions=0)
