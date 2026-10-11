"""任务轨迹模块（贯穿层：轨迹是 L0 反思与 L1 蒸馏的统一输入）。

「经验来自轨迹」：一条 Trajectory 记录 agent 完成一次任务
的完整步骤序列与最终验证结果，供 Reflector（L0）、ReasoningBank（L1）与
DataFlywheel（L2）消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Outcome:
    """轨迹结局。

    :param success: 是否成功。
    :param verifier: 判定来源（如 "rule"、"user"、"llm-judge"）。
    :param detail: 判定细节（失败原因、评分依据等）。
    """

    success: bool
    verifier: str
    detail: str = ""


@dataclass
class Trajectory:
    """一条任务轨迹。

    :param task: 任务描述（用户输入）。
    :param steps: 步骤文本序列，最后一个元素视为最终回答。
    :param outcome: 结局验证结果。
    """

    task: str
    steps: list[str] = field(default_factory=list)
    outcome: Outcome = field(default_factory=lambda: Outcome(False, "unknown"))

    @property
    def final_answer(self) -> str:
        """最终回答（步骤序列末尾）。"""
        return self.steps[-1] if self.steps else ""

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "task": self.task,
            "steps": list(self.steps),
            "outcome": {
                "success": self.outcome.success,
                "verifier": self.outcome.verifier,
                "detail": self.outcome.detail,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
        """从字典反序列化。"""
        o = d.get("outcome", {})
        return cls(
            task=d["task"],
            steps=list(d.get("steps", [])),
            outcome=Outcome(
                success=bool(o.get("success", False)),
                verifier=o.get("verifier", "unknown"),
                detail=o.get("detail", ""),
            ),
        )
