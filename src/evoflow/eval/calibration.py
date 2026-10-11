"""评委校准：LLM 评委上线前必须与人工标注对齐（Cohen's Kappa）。

设计红线："LLM 评委与人工一致率约 85% 是 GPT-4 评委的典型水平，上线前需人工校准"。
本模块提供纯标准库实现：
- :func:`cohens_kappa`：两标注者一致性系数（支持任意可哈希标签）。
- :func:`calibrate`：评委分数 vs 人工分数 → :class:`CalibrationReport`。
- :func:`sample_for_human`：分层随机抽样送人工标注（控制评测成本）。
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any


def cohens_kappa(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float:
    """Cohen's Kappa 一致性系数（纯标准库，支持任意可哈希标签）。

    :param labels_a: 标注者 A 的标签序列（如评委判分）。
    :param labels_b: 标注者 B 的标签序列（如人工标注）。
    :return: kappa ∈ [-1, 1]；1 完全一致，0 与随机一致相当，<0 不如随机。
    :raises ValueError: 长度不一致或为空。
    """
    a, b = list(labels_a), list(labels_b)
    if len(a) != len(b):
        raise ValueError(f"两序列长度不一致: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("空序列无法计算 kappa")
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if 1.0 - pe < 1e-12:
        # 退化情形：双方边际分布集中在同一标签，此时必有 po == 1
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


@dataclass
class CalibrationReport:
    """评委校准结果。

    :param kappa: Cohen's Kappa。
    :param agreement: 原始一致率（observed agreement）。
    :param passed: kappa 是否达到 min_kappa 阈值。
    :param confusion: 混淆计数，键为 ``(评委标签, 人工标签)``。
    :param suggestions: 未通过时的改进建议（通过时为空）。
    """

    kappa: float
    agreement: float
    passed: bool
    confusion: dict = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "agreement": self.agreement,
            "passed": self.passed,
            "confusion": {f"{a!r}|{b!r}": c for (a, b), c in self.confusion.items()},
            "suggestions": list(self.suggestions),
        }


_SUGGESTIONS = [
    "更换评委模型（如升级评委 LLM 或改用更强的判分模型）",
    "为评委提示词增加 few-shot 校准示例（覆盖当前混淆矩阵中的高频分歧对）",
    "提高人工复核比例（用 sample_for_human 把 ratio 上调，如 0.1 → 0.3）",
    "复核标签定义与评分细则，对分歧样本做双人标注后重测",
]


def calibrate(
    judge_scores: Sequence[Hashable],
    human_scores: Sequence[Hashable],
    min_kappa: float = 0.6,
) -> CalibrationReport:
    """评委分数与人工分数对齐校准。

    :param judge_scores: 评委（LLM）输出标签序列。
    :param human_scores: 人工标注标签序列（与 judge_scores 一一对应）。
    :param min_kappa: 上线阈值（默认 0.6，实质性一致）。
    :return: CalibrationReport；未过线时 passed=False 且附带改进建议。
    """
    judge, human = list(judge_scores), list(human_scores)
    if len(judge) != len(human):
        raise ValueError(f"两序列长度不一致: {len(judge)} vs {len(human)}")
    if not judge:
        raise ValueError("空序列无法校准")
    kappa = cohens_kappa(judge, human)
    agreement = sum(1 for x, y in zip(judge, human, strict=True) if x == y) / len(judge)
    confusion: dict = {}
    for x, y in zip(judge, human, strict=True):
        confusion[(x, y)] = confusion.get((x, y), 0) + 1
    passed = kappa >= min_kappa
    return CalibrationReport(
        kappa=kappa,
        agreement=agreement,
        passed=passed,
        confusion=confusion,
        suggestions=[] if passed else list(_SUGGESTIONS),
    )


def _strata_key(item: Any) -> Hashable:
    """推断分层键：dict 的 label/score/category 字段，或对象的同名属性。"""
    if isinstance(item, dict):
        for key in ("label", "score", "category"):
            if key in item:
                return item[key]
        return None
    for attr in ("label", "score", "category"):
        if hasattr(item, attr):
            return getattr(item, attr)
    return None


def sample_for_human(
    items: Sequence[Any], ratio: float = 0.1, seed: int = 42
) -> list[Any]:
    """分层随机抽样送人工标注（确定性：同 seed 同结果）。

    分层键自动推断（dict 的 ``label``/``score``/``category`` 或同名属性，
    推断不到时全体视为同一层）；每层抽样 ``max(1, round(层大小 * ratio))``
    条，保证稀有层至少有一条被人工覆盖。返回结果按原始顺序排列。

    :param items: 候选条目。
    :param ratio: 抽样比例（0, 1]，对应评测成本控制目标。
    :param seed: 随机种子（默认 42，可复现）。
    """
    items = list(items)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio 必须在 (0, 1] 区间内")
    strata: dict[Hashable, list[int]] = {}
    for idx, item in enumerate(items):
        strata.setdefault(_strata_key(item), []).append(idx)
    rng = random.Random(seed)
    picked: list[int] = []
    for key in sorted(strata, key=repr):  # 稳定遍历顺序，保证确定性
        idxs = strata[key]
        k = min(len(idxs), max(1, round(len(idxs) * ratio)))
        picked.extend(rng.sample(idxs, k))
    picked.sort()
    return [items[i] for i in picked]
