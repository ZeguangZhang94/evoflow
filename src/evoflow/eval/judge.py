"""评估门构件 —— Judge 评委协议与实现。

红线硬约束：**评委与 agent 运行时隔离**。评委不参与 agent 的生成过程、
不共享 agent 的可变状态；LLMJudge 通过独立 callable 包装外部评委，
从架构上防止"考生改卷子"。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Judge(Protocol):
    """评委协议：判定候选输出是否满足期望。"""

    def score(self, case_input: str, expected: str, output: str) -> bool:
        """返回 True 表示该用例通过。"""
        ...


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


class RuleBasedJudge:
    """零依赖规则评委（默认实现）。

    :param mode: ``contains``（期望答案出现在输出中）或 ``exact``（精确匹配）。
    :param case_sensitive: 是否大小写敏感（默认否）。
    """

    def __init__(self, mode: str = "contains", case_sensitive: bool = False) -> None:
        if mode not in ("contains", "exact"):
            raise ValueError("mode 必须是 'contains' 或 'exact'")
        self.mode = mode
        self.case_sensitive = case_sensitive

    def score(self, case_input: str, expected: str, output: str) -> bool:
        if self.case_sensitive:
            e, o = expected.strip(), output.strip()
        else:
            e, o = _normalize(expected), _normalize(output)
        if self.mode == "exact":
            return e == o
        return e in o


class LLMJudge:
    """外部评委包装器（评委隔离的执行点）。

    :param judge_fn: 独立的外部评委 callable，签名为
        ``(case_input, expected, output) -> bool``。它必须独立于被评 agent
        的运行时（独立进程 / 独立模型 / 独立上下文），本类只负责调用与
        结果布尔化，绝不把 agent 内部状态传入。
    """

    def __init__(self, judge_fn: Callable[[str, str, str], bool]) -> None:
        self._judge_fn = judge_fn

    def score(self, case_input: str, expected: str, output: str) -> bool:
        return bool(self._judge_fn(case_input, expected, output))
