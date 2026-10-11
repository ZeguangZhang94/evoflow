"""反馈信号采集模块（贯穿层：反馈信号采集）。

「贯穿：反馈信号采集」——用户的接受/拒绝/纠正是三层进化栈
（L0 上下文 / L1 记忆 / L2 权重）共同的原始燃料。信号统一落盘为 jsonl，
供数据飞轮（L2）构建偏好对、供 Reflector（L0）与 ReasoningBank（L1）消费。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class SignalType(str, Enum):
    """反馈信号类型。"""

    ACCEPT = "accept"        # 用户接受 agent 输出
    REJECT = "reject"        # 用户拒绝 agent 输出
    SUCCESS = "success"      # 任务被验证器判定成功
    FAIL = "fail"            # 任务被验证器判定失败
    CORRECTION = "correction"  # 用户给出纠正（飞轮中优先级最高）


@dataclass
class FeedbackSignal:
    """一条反馈信号。

    :param type: 信号类型。
    :param task_id: 关联的任务标识。
    :param value: 数值化强度（如评分 0~1，或 +1/-1）。
    :param meta: 附加信息；CORRECTION 信号约定包含 ``original`` / ``corrected`` 键。
    """

    type: SignalType
    task_id: str
    value: float
    meta: dict = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """序列化为可落盘字典。"""
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeedbackSignal:
        """从字典反序列化。"""
        d = dict(d)
        d["type"] = SignalType(d["type"])
        return cls(**d)


class SignalCollector:
    """信号收集器：内存累积 + 追加写 jsonl 落盘。

    :param path: jsonl 落盘路径；为 ``None`` 时仅保留在内存。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._signals: list[FeedbackSignal] = []
        if self.path is not None and self.path.exists():
            self._load()

    def collect(self, signal: FeedbackSignal) -> FeedbackSignal:
        """采集一条信号并立即追加落盘（保证崩溃不丢）。"""
        self._signals.append(signal)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(signal.to_dict(), ensure_ascii=False) + "\n")
        return signal

    def emit(
        self, type: SignalType, task_id: str, value: float, meta: dict | None = None
    ) -> FeedbackSignal:
        """便捷方法：构造并采集一条信号。"""
        return self.collect(
            FeedbackSignal(type=type, task_id=task_id, value=value, meta=meta or {})
        )

    def by_type(self, type: SignalType) -> list[FeedbackSignal]:
        """按类型过滤。"""
        return [s for s in self._signals if s.type is type]

    def _load(self) -> None:
        """从 jsonl 恢复历史信号。"""
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._signals.append(FeedbackSignal.from_dict(json.loads(line)))

    def __iter__(self) -> Iterator[FeedbackSignal]:
        return iter(self._signals)

    def __len__(self) -> int:
        return len(self._signals)
