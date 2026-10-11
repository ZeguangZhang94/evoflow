"""评估门测试：过线 PROMOTE / 不过线 REJECT / 带基线 ROLLBACK /
红线「不过线禁止晋升」/ 注册表「回滚恢复上一版」。"""

import pytest

from evoflow.core.gate import EvolutionGate, GateDecision, GateResult
from evoflow.eval.golden_set import GoldenCase, GoldenSet
from evoflow.eval.judge import LLMJudge, RuleBasedJudge
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied


def _golden() -> GoldenSet:
    return GoldenSet(
        [
            GoldenCase(input="重置密码", expected="设置"),
            GoldenCase(input="申请退款", expected="订单"),
        ]
    )


def _good(q: str) -> str:
    return {"重置密码": "请在设置里重置", "申请退款": "在订单页申请"}[q]


def _bad(q: str) -> str:
    return "不知道"


def test_pass_promotes():
    gate = EvolutionGate(_golden(), RuleBasedJudge(), min_pass_rate=0.9)
    result = gate.evaluate(_good)
    assert result.passed is True
    assert result.pass_rate == 1.0
    assert result.decision is GateDecision.PROMOTE
    assert result.failures == []


def test_fail_rejects_without_baseline():
    gate = EvolutionGate(_golden(), RuleBasedJudge(), min_pass_rate=0.9)
    result = gate.evaluate(_bad)
    assert result.passed is False
    assert result.pass_rate == 0.0
    assert result.decision is GateDecision.REJECT
    assert len(result.failures) == 2


def test_fail_rollbacks_with_baseline():
    gate = EvolutionGate(_golden(), RuleBasedJudge(), min_pass_rate=0.9, baseline=_good)
    result = gate.evaluate(_bad)
    assert result.passed is False
    assert result.decision is GateDecision.ROLLBACK


def test_empty_golden_set_refused():
    with pytest.raises(ValueError):
        EvolutionGate(GoldenSet([]), RuleBasedJudge())


def test_invalid_min_pass_rate():
    with pytest.raises(ValueError):
        EvolutionGate(_golden(), RuleBasedJudge(), min_pass_rate=1.5)


def test_llm_judge_wraps_external_callable():
    judge = LLMJudge(lambda i, e, o: e in o)
    gate = EvolutionGate(_golden(), judge)
    assert gate.evaluate(_good).passed is True


def test_redline_no_promotion_below_line(tmp_path):
    """红线：评估门不过线，注册表必须拒绝晋升。"""
    registry = AdapterRegistry(tmp_path)
    version = registry.register(tmp_path / "adapter_v1", metrics={"pass_rate": 0.0})
    gate = EvolutionGate(_golden(), RuleBasedJudge(), min_pass_rate=0.9)
    bad_result = gate.evaluate(_bad)
    with pytest.raises(PromotionDenied):
        registry.promote(version, bad_result)
    assert registry.current is None  # 未被污染


def test_rollback_restores_previous_version(tmp_path):
    """晋升 v2 后回滚，必须恢复 v1。"""
    registry = AdapterRegistry(tmp_path)
    passed = GateResult(passed=True, pass_rate=1.0, decision=GateDecision.PROMOTE)
    v1 = registry.register(tmp_path / "a1", metrics={})
    v2 = registry.register(tmp_path / "a2", metrics={})
    registry.promote(v1, passed)
    registry.promote(v2, passed)
    assert registry.current == v2
    assert registry.rollback() == v1
    assert registry.current == v1
    assert registry.rollback() is None  # 无更多历史
    assert registry.current == v1  # 保持不动
