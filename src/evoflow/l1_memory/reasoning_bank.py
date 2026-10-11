"""L1 经验/记忆库 —— ReasoningBank（类 ReasoningBank 的策略蒸馏与检索）。

对应 ReasoningBank（arXiv:2509.25140，详见 docs/research-to-code.md）：从**成功和失败**
两类轨迹中蒸馏结构化策略条目（StrategyItem），嵌入入库；推理时按查询检索注入，
默认 k=1——原论文强调「相关性 > 数量」，注入过多条目只会稀释上下文。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from evoflow.core.trajectory import Trajectory
from evoflow.l1_memory.store import EmbedFn, VectorStore


@dataclass
class StrategyItem:
    """一条蒸馏出的策略条目。

    :param title: 条目标题。
    :param description: 一句话描述（适用场景）。
    :param content: 策略正文（可执行的做法/教训）。
    """

    title: str
    description: str
    content: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StrategyItem:
        return cls(**d)

    def searchable_text(self) -> str:
        """用于嵌入检索的拼接文本。"""
        return f"{self.title}\n{self.description}\n{self.content}"


def _default_distill(trajectory: Trajectory) -> StrategyItem:
    """内置规则蒸馏器：成功提炼做法，失败提炼教训。"""
    task_short = trajectory.task[:30]
    steps_text = " -> ".join(trajectory.steps[:5])
    if trajectory.outcome.success:
        return StrategyItem(
            title=f"成功策略：{task_short}",
            description=(
                f"任务「{task_short}」的可复用做法（验证器: {trajectory.outcome.verifier}）"
            ),
            content=(
                f"任务：{trajectory.task}\n有效步骤：{steps_text}\n"
                f"结论：{trajectory.outcome.detail or '按此步骤可成功'}"
            ),
            meta={"kind": "success"},
        )
    return StrategyItem(
        title=f"失败教训：{task_short}",
        description=f"任务「{task_short}」的避坑指南（验证器: {trajectory.outcome.verifier}）",
        content=f"任务：{trajectory.task}\n失败步骤：{steps_text}\n失败原因：{trajectory.outcome.detail}\n教训：避免重蹈上述做法",
        meta={"kind": "fail"},
    )


class ReasoningBank:
    """经验记忆库：蒸馏 + 检索。

    :param store: 向量存储（实现 :class:`VectorStore` 协议）。
    :param embed_fn: 可选嵌入函数；提供时会覆盖 store 的 embed_fn。
    :param distill_fn: 可插拔蒸馏函数；缺省用内置规则蒸馏器。
    """

    def __init__(
        self,
        store: VectorStore,
        embed_fn: EmbedFn | None = None,
        distill_fn: Callable[[Trajectory], StrategyItem] | None = None,
    ) -> None:
        if embed_fn is not None:
            store.embed_fn = embed_fn  # type: ignore[attr-defined]
        self.store = store
        self._distill_fn = distill_fn or _default_distill

    def distill(self, trajectory: Trajectory) -> StrategyItem:
        """蒸馏一条轨迹（成功/失败都蒸馏）并入库。"""
        item = self._distill_fn(trajectory)
        self.store.add(
            item_id=f"s-{uuid.uuid4().hex[:12]}",
            text=item.searchable_text(),
            payload=item.to_dict(),
        )
        return item

    def retrieve(self, query: str, k: int = 1) -> list[StrategyItem]:
        """按查询检索最相关的策略条目。

        默认 k=1：原论文原则「相关性 > 数量」，宁缺毋滥。
        """
        return [StrategyItem.from_dict(p) for p, _ in self.store.search(query, k=k)]

    def __len__(self) -> int:
        return len(self.store)  # type: ignore[arg-type]
