"""评估门模块（贯穿层：评估门 —— 冻结黄金集回归测试）。

红线硬约束：任何候选（新 playbook、新 adapter 版本）必须通过评估门才允许
晋升；不过线一律拒绝。黄金集冻结、评委独立，防止"自己考自己"。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from evoflow.eval.golden_set import GoldenSet
from evoflow.eval.judge import Judge


class GateDecision(str, Enum):
    """评估门裁决。"""

    PROMOTE = "promote"    # 过线：允许晋升
    REJECT = "reject"      # 不过线且无旧版本：拒绝候选
    ROLLBACK = "rollback"  # 不过线但存在已晋升基线：回滚到基线版本


@dataclass
class GateResult:
    """评估门裁决结果。

    :param passed: 是否过线。
    :param pass_rate: 黄金集通过率。
    :param failures: 未通过用例明细（input / expected / output）。
    :param decision: 裁决。
    """

    passed: bool
    pass_rate: float
    failures: list[dict] = field(default_factory=list)
    decision: GateDecision = GateDecision.REJECT

    def to_dict(self) -> dict:
        """序列化（供注册表留存）。"""
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "failures": self.failures,
            "decision": self.decision.value,
        }


class EvolutionGate:
    """评估门：用冻结黄金集对候选 callable 做回归测试。

    :param golden_set: 冻结的黄金集。
    :param judge: 评委（与 agent 运行时隔离，见 :mod:`evoflow.eval.judge`）。
    :param min_pass_rate: 过线阈值，默认 0.9。
    :param baseline: 可选的当前已晋升版本 callable；候选不过线且存在基线时
        裁决为 ROLLBACK，否则为 REJECT。
    """

    def __init__(
        self,
        golden_set: GoldenSet,
        judge: Judge,
        min_pass_rate: float = 0.9,
        baseline: Callable[[str], str] | None = None,
    ) -> None:
        if not 0.0 < min_pass_rate <= 1.0:
            raise ValueError("min_pass_rate 必须在 (0, 1] 区间内")
        if len(golden_set) == 0:
            raise ValueError("黄金集为空，评估门拒绝工作（防止无测试晋升）")
        self.golden_set = golden_set
        self.judge = judge
        self.min_pass_rate = min_pass_rate
        self.baseline = baseline

    def evaluate(self, candidate: Callable[[str], str]) -> GateResult:
        """评估候选并给出裁决。

        :param candidate: 候选 callable，输入题目文本、输出回答文本。
        :return: GateResult。
        """
        total = 0
        passed = 0
        failures: list[dict] = []
        for case, output in self.golden_set.run(candidate):
            total += 1
            if self.judge.score(case.input, case.expected, output):
                passed += 1
            else:
                failures.append(
                    {"input": case.input, "expected": case.expected, "output": output}
                )
        pass_rate = passed / total if total else 0.0
        ok = pass_rate >= self.min_pass_rate
        if ok:
            decision = GateDecision.PROMOTE
        elif self.baseline is not None:
            decision = GateDecision.ROLLBACK
        else:
            decision = GateDecision.REJECT
        return GateResult(
            passed=ok, pass_rate=pass_rate, failures=failures, decision=decision
        )
