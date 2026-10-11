"""L2 权重层 —— 数据飞轮：反馈信号 -> 偏好对 -> DPO 训练数据。

L2 层「数据飞轮」设计（详见 docs/research-to-code.md）：
1. **用户纠正优先**：CORRECTION 信号中用户亲手给出的 corrected/original
   是最高质量偏好对，无条件优先采用。
2. **同任务成败配对**：同一 task 下成功轨迹为 chosen、失败轨迹为 rejected。
3. **去重**：(task, chosen, rejected) 规范化后重复的对只保留一份。
4. **比例控制**：pos_neg_ratio 表示每个任务下 正:负 样本的目标比例（默认 2.0，
   即正样本数量约为负样本的 2 倍，数据飞轮实践中的常用配比「正负样本比 2:1」），
   防止负样本淹没训练分布。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from evoflow.core.signals import FeedbackSignal, SignalCollector, SignalType
from evoflow.core.trajectory import Trajectory


@dataclass
class PreferencePair:
    """一条 DPO 偏好对。

    :param chosen: 偏好（更优）的回答。
    :param rejected: 被拒绝的回答。
    :param task: 任务描述（训练时作为 prompt）。
    """

    chosen: str
    rejected: str
    task: str

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


class DataFlywheel:
    """数据飞轮：把信号与轨迹碾成偏好对。

    :param collector: 信号收集器（``build_pairs`` 未显式传 signals 时使用）。
    """

    def __init__(self, collector: SignalCollector | None = None) -> None:
        self.collector = collector
        self._last_pairs: list[PreferencePair] = []

    def build_pairs(
        self,
        signals: Iterable[FeedbackSignal] | None = None,
        trajectories: Iterable[Trajectory] | None = None,
        pos_neg_ratio: float = 2.0,
    ) -> list[PreferencePair]:
        """构建偏好对。

        :param signals: 反馈信号；缺省取自 collector。
        :param trajectories: 任务轨迹。
        :param pos_neg_ratio: 每个任务的 正:负 目标比例（默认 2.0，正多于负）。
        :return: 去重后的偏好对列表（纠正对在前）。
        """
        if signals is None:
            if self.collector is None:
                raise ValueError("未提供 signals 且 flywheel 未绑定 collector")
            signals = list(self.collector)
        trajectories = list(trajectories or [])

        pairs: list[PreferencePair] = []
        seen: set[tuple[str, str, str]] = set()

        def _push(chosen: str, rejected: str, task: str) -> None:
            if not chosen or not rejected or _norm(chosen) == _norm(rejected):
                return
            key = (_norm(task), _norm(chosen), _norm(rejected))
            if key in seen:
                return
            seen.add(key)
            pairs.append(PreferencePair(chosen=chosen, rejected=rejected, task=task))

        # 规则 1：用户纠正优先
        for s in signals:
            if s.type is SignalType.CORRECTION:
                corrected = str(s.meta.get("corrected", ""))
                original = str(s.meta.get("original", ""))
                task = str(s.meta.get("task", s.task_id))
                _push(chosen=corrected, rejected=original, task=task)

        # 规则 2：同任务成败配对（带比例控制）
        by_task: dict[str, dict[str, list[str]]] = {}
        for t in trajectories:
            bucket = by_task.setdefault(t.task, {"chosen": [], "rejected": []})
            answer = t.final_answer
            if not answer:
                continue
            bucket["chosen" if t.outcome.success else "rejected"].append(answer)

        for task, bucket in by_task.items():
            chosens = bucket["chosen"]
            rejecteds = bucket["rejected"]
            if not chosens or not rejecteds:
                continue
            max_rejected = max(1, int(len(chosens) / pos_neg_ratio))
            for i, rejected in enumerate(rejecteds[:max_rejected]):
                _push(chosen=chosens[i % len(chosens)], rejected=rejected, task=task)

        self._last_pairs = pairs
        return pairs

    def export_dpo_jsonl(
        self, path: str | Path, pairs: Iterable[PreferencePair] | None = None
    ) -> int:
        """导出为 DPO 训练用 jsonl（prompt / chosen / rejected 三字段）。

        :param path: 输出路径。
        :param pairs: 要导出的偏好对；缺省使用最近一次 :meth:`build_pairs` 的结果。
        """
        pairs = list(pairs) if pairs is not None else self._last_pairs
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                row = {"prompt": p.task, "chosen": p.chosen, "rejected": p.rejected}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        return n
