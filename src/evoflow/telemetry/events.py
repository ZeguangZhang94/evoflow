"""进化事件：EvoFlow 可观测性的最小数据单元。

每一次信号采集、反思、蒸馏、偏好对生成、训练轮、评估门裁决、晋升、
回滚、熔断都产生一个 :class:`EvolutionEvent`，追加进事件流水后，
即可回放整条进化轨迹（playbook 增长曲线、黄金集通过率曲线、
adapter 版本树等）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class EventKind(str, Enum):
    """进化事件类别。"""

    SIGNAL = "signal"            # 反馈信号采集
    REFLECT = "reflect"          # L0 Reflector 反思
    DISTILL = "distill"          # L1 经验蒸馏
    PAIR_BUILT = "pair_built"    # L2 偏好对生成
    TRAIN_ROUND = "train_round"  # 一轮自训练完成
    GATE_RESULT = "gate_result"  # 评估门裁决
    PROMOTE = "promote"          # adapter / playbook 晋升
    ROLLBACK = "rollback"        # 回滚到基线
    HALT = "halt"                # 熔断 / 红线拦截


@dataclass
class EvolutionEvent:
    """一条进化事件。

    :param kind: 事件类别（接受 :class:`EventKind` 或其字符串值，未知值抛错）。
    :param payload: 事件载荷（必须可 JSON 序列化；指标约定放在
        ``payload["metrics"]`` 字典或顶层键中，供
        :meth:`EvolutionHistory.metrics_series` 提取）。
    :param ts: Unix 时间戳，默认当前时刻。
    """

    kind: EventKind | str
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventKind):
            try:
                self.kind = EventKind(self.kind)
            except ValueError as e:
                valid = ", ".join(k.value for k in EventKind)
                raise ValueError(f"未知事件类别: {self.kind!r}（合法值: {valid}）") from e
        if not isinstance(self.payload, dict):
            raise TypeError("payload 必须是 dict")
        self.ts = float(self.ts)

    def to_dict(self) -> dict:
        """序列化为 JSON 友好字典。"""
        assert isinstance(self.kind, EventKind)  # __post_init__ 已校正为 EventKind
        return {"kind": self.kind.value, "payload": self.payload, "ts": self.ts}

    @classmethod
    def from_dict(cls, row: dict) -> EvolutionEvent:
        """从字典反序列化（jsonl 读取用）。"""
        return cls(kind=row["kind"], payload=row.get("payload", {}), ts=row["ts"])
