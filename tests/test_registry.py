"""L2 注册表测试：版本登记、红线晋升约束、当前版本指针持久化。"""

import pytest

from evoflow.core.gate import GateDecision, GateResult
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied

PASSED = GateResult(passed=True, pass_rate=0.95, decision=GateDecision.PROMOTE)
FAILED = GateResult(passed=False, pass_rate=0.5, decision=GateDecision.REJECT)


def test_register_increments_version(tmp_path):
    registry = AdapterRegistry(tmp_path)
    v1 = registry.register(tmp_path / "a1", metrics={"loss": 0.3})
    v2 = registry.register(tmp_path / "a2", metrics={"loss": 0.2})
    assert (v1, v2) == ("v0001", "v0002")
    assert registry.versions() == ["v0001", "v0002"]
    assert registry.info(v1)["metrics"] == {"loss": 0.3}


def test_promote_requires_passed_gate(tmp_path):
    """红线：无评估门结果或结果未通过，一律拒绝晋升。"""
    registry = AdapterRegistry(tmp_path)
    v1 = registry.register(tmp_path / "a1", metrics={})
    with pytest.raises(PromotionDenied):
        registry.promote(v1)  # 无 gate 结果
    with pytest.raises(PromotionDenied):
        registry.promote(v1, FAILED)  # gate 未通过
    assert registry.current is None


def test_promote_uses_stored_gate_result(tmp_path):
    registry = AdapterRegistry(tmp_path)
    v1 = registry.register(tmp_path / "a1", metrics={}, gate_result=PASSED)
    assert registry.promote(v1) == v1  # 登记时留存的结果即可晋升
    assert registry.current == v1
    assert registry.current_path() == tmp_path / "a1"


def test_promote_unknown_version(tmp_path):
    registry = AdapterRegistry(tmp_path)
    with pytest.raises(KeyError):
        registry.promote("v9999", PASSED)


def test_current_pointer_persists_across_instances(tmp_path):
    registry = AdapterRegistry(tmp_path)
    v1 = registry.register(tmp_path / "a1", metrics={})
    registry.promote(v1, PASSED)
    # 新实例从磁盘恢复
    registry2 = AdapterRegistry(tmp_path)
    assert registry2.current == v1
    assert registry2.rollback() is None
