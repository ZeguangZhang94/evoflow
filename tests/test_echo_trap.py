"""Echo Trap 监控器测试：三个触发信号、正常情况、enforce 熔断。"""

import pytest

from evoflow.l2_weight.trainers.echo_trap import (
    EchoTrapHalted,
    EchoTrapMonitor,
    EchoTrapStatus,
)


def _healthy_records(m: EchoTrapMonitor, n: int) -> None:
    for i in range(n):
        m.record(
            rewards=[1.0, 2.0, 3.0, 4.0],
            entropy=1.0,
            grad_norm=1.0 + 0.01 * i,
        )


class TestNormalCase:
    def test_healthy_window_is_ok(self):
        m = EchoTrapMonitor(window=4)
        _healthy_records(m, 4)
        status = m.check()
        assert status.ok
        assert status.signals == []

    def test_enforce_passes_when_healthy(self):
        m = EchoTrapMonitor(window=4)
        _healthy_records(m, 4)
        assert m.enforce().ok

    def test_cold_start_window_not_full_no_variance_signal(self):
        """窗口未满时，即使方差为 0 也不报方差塌缩（避免冷启动误报）。"""
        m = EchoTrapMonitor(window=8)
        for _ in range(3):
            m.record(rewards=[0.5, 0.5, 0.5], entropy=1.0, grad_norm=1.0)
        assert m.check().ok

    def test_record_requires_two_rewards(self):
        m = EchoTrapMonitor()
        with pytest.raises(ValueError):
            m.record(rewards=[1.0], entropy=1.0, grad_norm=1.0)


class TestRewardVarianceCollapse:
    def test_triggers_when_window_variance_below_floor(self):
        m = EchoTrapMonitor(window=4, var_floor=1e-3)
        for _ in range(4):
            m.record(rewards=[0.5, 0.5, 0.5], entropy=1.0, grad_norm=1.0)
        status = m.check()
        assert not status.ok
        assert any("方差" in s for s in status.signals)

    def test_not_triggered_when_rewards_diverse(self):
        m = EchoTrapMonitor(window=4, var_floor=1e-3)
        _healthy_records(m, 4)  # pvariance([1,2,3,4]) = 1.25 >> floor
        assert m.check().ok


class TestEntropyCollapse:
    def test_triggers_when_below_floor_and_declining(self):
        m = EchoTrapMonitor(window=8, entropy_floor=0.1)
        for e in (0.09, 0.06, 0.03):
            m.record(rewards=[0.0, 1.0], entropy=e, grad_norm=1.0)
        status = m.check()
        assert not status.ok
        assert any("熵" in s for s in status.signals)

    def test_declining_but_above_floor_is_ok(self):
        m = EchoTrapMonitor(window=8, entropy_floor=0.1)
        for e in (1.0, 0.9, 0.8):
            m.record(rewards=[0.0, 1.0], entropy=e, grad_norm=1.0)
        assert m.check().ok

    def test_below_floor_but_recovering_is_ok(self):
        m = EchoTrapMonitor(window=8, entropy_floor=0.1)
        for e in (0.05, 0.04, 0.06):  # 末段回升，非持续下降
            m.record(rewards=[0.0, 1.0], entropy=e, grad_norm=1.0)
        assert m.check().ok


class TestGradSpike:
    def test_triggers_when_spike_exceeds_multiplier(self):
        m = EchoTrapMonitor(window=8, grad_spike=5.0)
        for _ in range(3):
            m.record(rewards=[0.0, 1.0], entropy=1.0, grad_norm=1.0)
        m.record(rewards=[0.0, 1.0], entropy=1.0, grad_norm=10.0)
        status = m.check()
        assert not status.ok
        assert any("梯度" in s for s in status.signals)

    def test_moderate_increase_is_ok(self):
        m = EchoTrapMonitor(window=8, grad_spike=5.0)
        for _ in range(3):
            m.record(rewards=[0.0, 1.0], entropy=1.0, grad_norm=1.0)
        m.record(rewards=[0.0, 1.0], entropy=1.0, grad_norm=4.0)  # < 5x
        assert m.check().ok


class TestEnforce:
    def test_raises_with_signal_details(self):
        m = EchoTrapMonitor(window=4)
        for _ in range(4):
            m.record(rewards=[0.5, 0.5], entropy=1.0, grad_norm=1.0)
        with pytest.raises(EchoTrapHalted, match="方差塌缩"):
            m.enforce()

    def test_status_dataclass_shape(self):
        status = EchoTrapStatus(ok=True)
        assert status.ok and status.signals == []
