"""评估门构件 —— 冻结黄金集（GoldenSet）。

评估门设计：黄金集是冻结的回归测试集（字段 input / expected），
任何进化产物晋升前都必须重跑全量用例。黄金集一经冻结不允许在进化循环
内被修改——防止"为了过线而改题"。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenCase:
    """一条黄金用例。

    :param input: 题目输入。
    :param expected: 期望答案（评委判定基准）。
    :param meta: 附加信息（来源、难度等）。
    """

    input: str
    expected: str
    meta: dict = field(default_factory=dict)


class GoldenSet:
    """冻结黄金集。"""

    def __init__(self, cases: list[GoldenCase]) -> None:
        self.cases = list(cases)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GoldenSet:
        """从 jsonl 加载（每行至少含 input / expected 字段）。"""
        cases: list[GoldenCase] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "input" not in row or "expected" not in row:
                    raise ValueError(f"{path}:{lineno} 缺少 input/expected 字段")
                cases.append(
                    GoldenCase(
                        input=row["input"],
                        expected=row["expected"],
                        meta={k: v for k, v in row.items() if k not in ("input", "expected")},
                    )
                )
        return cls(cases)

    def run(
        self, candidate: Callable[[str], str]
    ) -> Iterator[tuple[GoldenCase, str]]:
        """遍历黄金集：对每条用例调用候选并产出 (case, output)。"""
        for case in self.cases:
            yield case, candidate(case.input)

    def __iter__(self) -> Iterator[GoldenCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)
