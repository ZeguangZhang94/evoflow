"""进化历史流水：jsonl 追加式事件存储 + 查询 + 组合式插桩。

设计原则：**组合优于继承**。``wrap_gate`` / ``wrap_registry`` 返回包装
对象，拦截 ``evaluate`` / ``promote`` / ``rollback`` 自动记录事件，
不修改 :class:`EvolutionGate` / :class:`AdapterRegistry` 原类的一行代码；
未拦截的属性/方法通过 ``__getattr__`` 原样透传，包装对象可无缝替换原对象。
"""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path

from evoflow.core.gate import EvolutionGate, GateDecision
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied
from evoflow.telemetry.events import EventKind, EvolutionEvent


class EvolutionHistory:
    """进化事件流水（jsonl 追加式）。

    :param path: jsonl 文件路径（父目录自动创建；文件不存在时视为空流水）。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 写入 ----------

    def record(self, event: EvolutionEvent) -> EvolutionEvent:
        """追加一条事件（一行 JSON）。"""
        if not isinstance(event, EvolutionEvent):
            raise TypeError("record() 只接受 EvolutionEvent")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def emit(
        self,
        kind: EventKind | str,
        payload: dict | None = None,
        ts: float | None = None,
    ) -> EvolutionEvent:
        """构造并记录一条事件的便捷方法。"""
        event = EvolutionEvent(kind=kind, payload=payload or {})
        if ts is not None:
            event.ts = float(ts)
        return self.record(event)

    # ---------- 读取 ----------

    def query(
        self,
        kind: EventKind | str | None = None,
        since: float | None = None,
    ) -> list[EvolutionEvent]:
        """按类别 / 起始时间过滤事件，按 ts 升序返回。

        :param kind: 只保留该类别（EventKind 或其字符串值）；None 不过滤。
        :param since: 只保留 ``ts >= since`` 的事件；None 不过滤。
        """
        if kind is not None and not isinstance(kind, EventKind):
            kind = EventKind(kind)
        events: list[EvolutionEvent] = []
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = EvolutionEvent.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    raise ValueError(f"{self.path}:{lineno} 事件行损坏: {e}") from e
                if kind is not None and event.kind is not kind:
                    continue
                if since is not None and event.ts < since:
                    continue
                events.append(event)
        events.sort(key=lambda e: e.ts)
        return events

    def metrics_series(self, metric_name: str) -> list[tuple[float, float]]:
        """提取某指标的 (ts, value) 时间序列（供画增长/通过率曲线）。

        查找顺序：``payload[metric_name]`` → ``payload["metrics"][metric_name]``；
        只收集数值（int/float，排除 bool）项。
        """
        series: list[tuple[float, float]] = []
        for event in self.query():
            value = event.payload.get(metric_name)
            if value is None:
                metrics = event.payload.get("metrics")
                if isinstance(metrics, dict):
                    value = metrics.get(metric_name)
            if isinstance(value, Real) and not isinstance(value, bool):
                series.append((event.ts, float(value)))
        return series


# ----------------------------------------------------------------------
# 组合式插桩（不修改原类）
# ----------------------------------------------------------------------


class _InstrumentedGate:
    """EvolutionGate 包装器：evaluate 自动记录 GATE_RESULT 事件。

    裁决为 PROMOTE / ROLLBACK 时额外记录对应事件，回放时可直接看到
    "评估 → 晋升 / 回滚"的完整链路。
    """

    def __init__(self, gate: EvolutionGate, history: EvolutionHistory) -> None:
        self._gate = gate
        self._history = history

    @property
    def wrapped(self) -> EvolutionGate:
        """被包装的原对象。"""
        return self._gate

    def evaluate(self, candidate):
        result = self._gate.evaluate(candidate)
        self._history.emit(
            EventKind.GATE_RESULT,
            {
                "passed": result.passed,
                "decision": result.decision.value,
                "failures_count": len(result.failures),
                "metrics": {"gate.pass_rate": result.pass_rate},
            },
        )
        if result.decision is GateDecision.PROMOTE:
            self._history.emit(EventKind.PROMOTE, {"via": "gate"})
        elif result.decision is GateDecision.ROLLBACK:
            self._history.emit(EventKind.ROLLBACK, {"via": "gate"})
        return result

    def __getattr__(self, name: str):
        # 只会在实例属性找不到时触发（_gate/_history 已在 __dict__），安全透传
        return getattr(self._gate, name)


class _InstrumentedRegistry:
    """AdapterRegistry 包装器：promote / rollback 自动记录事件。

    红线语义：promote 被 :class:`PromotionDenied` 拦截时记录 HALT 事件
    （红线拦截属于安全停机），然后把原异常继续抛出。
    """

    def __init__(self, registry: AdapterRegistry, history: EvolutionHistory) -> None:
        self._registry = registry
        self._history = history

    @property
    def wrapped(self) -> AdapterRegistry:
        """被包装的原对象。"""
        return self._registry

    def promote(self, version: str, gate_result=None) -> str:
        try:
            promoted = self._registry.promote(version, gate_result)
        except PromotionDenied as e:
            self._history.emit(
                EventKind.HALT, {"reason": "promotion_denied", "version": version, "detail": str(e)}
            )
            raise
        payload: dict = {"version": promoted, "current": self._registry.current}
        gr = gate_result if gate_result is not None else None
        if gr is not None and getattr(gr, "pass_rate", None) is not None:
            payload["metrics"] = {"gate.pass_rate": float(gr.pass_rate)}
        self._history.emit(EventKind.PROMOTE, payload)
        return promoted

    def rollback(self):
        restored = self._registry.rollback()
        if restored is not None:
            self._history.emit(
                EventKind.ROLLBACK, {"via": "registry", "restored": restored}
            )
        return restored

    def __getattr__(self, name: str):
        return getattr(self._registry, name)


def wrap_gate(gate: EvolutionGate, history: EvolutionHistory) -> _InstrumentedGate:
    """包装评估门：evaluate 自动落事件（组合，不改原类）。"""
    return _InstrumentedGate(gate, history)


def wrap_registry(
    registry: AdapterRegistry, history: EvolutionHistory
) -> _InstrumentedRegistry:
    """包装注册表：promote / rollback 自动落事件（组合，不改原类）。"""
    return _InstrumentedRegistry(registry, history)
