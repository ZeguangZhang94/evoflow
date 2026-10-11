"""L0 上下文进化 —— Reflector（类 ACE 的反思者）。

对应 ACE 的 Generator -> Reflector -> Curator 链路（详见 docs/research-to-code.md）：Reflector 消费
任务轨迹，产出若干条文本洞察，包装为 DeltaEntry 交给 Playbook.curate。
反思函数 ``reflect_fn`` 可插拔（规则、模板或外部 LLM），核心包不依赖任何
第三方 SDK。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from evoflow.core.trajectory import Trajectory
from evoflow.l0_context.playbook import DeltaEntry


class Reflector:
    """反思者：从轨迹提炼增量条目。

    :param reflect_fn: 可插拔反思函数，输入轨迹、输出洞察文本列表。
    """

    def __init__(self, reflect_fn: Callable[[Trajectory], list[str]]) -> None:
        self._reflect_fn = reflect_fn

    def reflect(self, trajectory: Trajectory) -> list[DeltaEntry]:
        """对一条轨迹反思，返回 DeltaEntry 列表（未入库，交由 Playbook.curate）。

        每条洞察自动打上 ``reflect`` 与 ``success``/``fail`` 标签。
        """
        insights = self._reflect_fn(trajectory)
        outcome_tag = "success" if trajectory.outcome.success else "fail"
        return [
            DeltaEntry(
                id=f"d-{uuid.uuid4().hex[:12]}",
                content=text,
                tags=["reflect", outcome_tag],
            )
            for text in insights
            if text and text.strip()
        ]
