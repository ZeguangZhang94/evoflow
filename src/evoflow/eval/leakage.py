"""黄金集防泄漏检查（红线：黄金集绝不能进训练数据）。

飞轮导出训练对之后、``train()`` 之前必须跑本检查：
- **精确匹配**：规范化（去空白/统一小写）后逐字段比对；
- **近重复**：char n-gram Jaccard 相似度 > 阈值（默认 0.8），纯标准库实现。

黄金集取 ``input`` / ``expected`` 两字段；训练对（PreferencePair）查
``task`` / ``chosen`` / ``rejected`` 三个字段——任何一处命中都算泄漏。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class LeakageError(RuntimeError):
    """黄金集泄漏：训练数据中检出黄金集内容（精确或近重复）。"""

    def __init__(self, message: str, report: LeakageReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class LeakageReport:
    """泄漏检查结果。

    :param leaked: 命中明细，每条为 dict：
        ``golden_index`` / ``golden_field`` / ``train_index`` /
        ``train_field`` / ``match``（``exact`` 或 ``near``）/ ``similarity``。
    :param ok: 无泄漏为 True。
    """

    leaked: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.leaked


def _normalize(text: str) -> str:
    """规范化：折叠空白 + 小写（与 flywheel 的 _norm 口径一致）。"""
    return " ".join(str(text).split()).lower()


def _ngrams(text: str, n: int) -> set[str]:
    """char n-gram 集合；文本短于 n 时退化为整串单元素集合。"""
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _field(obj: Any, name: str) -> str:
    """同时支持 dataclass 属性与 dict 键的字段读取。"""
    if isinstance(obj, dict):
        return str(obj.get(name, ""))
    return str(getattr(obj, name, ""))


_GOLDEN_FIELDS = ("input", "expected")
_TRAIN_FIELDS = ("task", "chosen", "rejected")


def check_leakage(
    golden_set: Iterable,
    train_pairs: Iterable,
    ngram: int = 3,
    threshold: float = 0.8,
) -> LeakageReport:
    """检查黄金集是否泄漏进训练数据（精确 + 近重复双重检测）。

    :param golden_set: GoldenSet（或 GoldenCase 可迭代，取 input/expected）。
    :param train_pairs: 飞轮的 PreferencePair 可迭代（查 task/chosen/rejected）。
    :param ngram: char n-gram 的 n（默认 3）。
    :param threshold: 近重复 Jaccard 阈值（默认 0.8，严格大于判泄漏）。
    :return: LeakageReport。
    """
    goldens = list(golden_set)
    pairs = list(train_pairs)
    golden_norm: list[tuple[int, str, str]] = []  # (index, field, normalized)
    for gi, case in enumerate(goldens):
        for f in _GOLDEN_FIELDS:
            golden_norm.append((gi, f, _normalize(_field(case, f))))
    golden_grams = [(gi, f, _ngrams(t, ngram)) for gi, f, t in golden_norm]

    report = LeakageReport()
    for ti, pair in enumerate(pairs):
        for tf in _TRAIN_FIELDS:
            text = _normalize(_field(pair, tf))
            if not text:
                continue
            grams = _ngrams(text, ngram)
            for (gi, gf, gnorm), (_, _, ggrams) in zip(
                golden_norm, golden_grams, strict=True
            ):
                if not gnorm:
                    continue
                if text == gnorm:
                    report.leaked.append(
                        {
                            "golden_index": gi,
                            "golden_field": gf,
                            "train_index": ti,
                            "train_field": tf,
                            "match": "exact",
                            "similarity": 1.0,
                        }
                    )
                else:
                    sim = _jaccard(grams, ggrams)
                    if sim > threshold:
                        report.leaked.append(
                            {
                                "golden_index": gi,
                                "golden_field": gf,
                                "train_index": ti,
                                "train_field": tf,
                                "match": "near",
                                "similarity": sim,
                            }
                        )
    return report


def assert_no_leakage(
    golden_set: Iterable,
    train_pairs: Iterable,
    ngram: int = 3,
    threshold: float = 0.8,
) -> LeakageReport:
    """泄漏即抛 :class:`LeakageError`（供飞轮导出后、train() 前调用）。

    :return: 无泄漏时返回 LeakageReport（ok=True）。
    """
    report = check_leakage(golden_set, train_pairs, ngram=ngram, threshold=threshold)
    if not report.ok:
        first = report.leaked[0]
        raise LeakageError(
            f"红线触发：检测到黄金集泄漏 {len(report.leaked)} 处，禁止训练。"
            f"首例: golden#{first['golden_index']}.{first['golden_field']} vs "
            f"train#{first['train_index']}.{first['train_field']} "
            f"({first['match']}, similarity={first['similarity']:.3f})",
            report,
        )
    return report
