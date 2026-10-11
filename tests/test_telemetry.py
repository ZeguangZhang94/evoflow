"""进化可观测性测试：事件记录/查询/指标序列、wrap_gate/wrap_registry
组合插桩、exporter 惰性依赖报错与增量导出（假模块模拟，零第三方依赖）。"""

import sys
import types

import pytest

from evoflow.core.gate import EvolutionGate, GateDecision, GateResult
from evoflow.eval.golden_set import GoldenCase, GoldenSet
from evoflow.eval.judge import RuleBasedJudge
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied
from evoflow.telemetry import (
    EventKind,
    EvolutionEvent,
    EvolutionHistory,
    LangfuseExporter,
    OpenTelemetryExporter,
    wrap_gate,
    wrap_registry,
)

# ---------- 事件与历史流水 ----------


def test_event_kind_coercion_and_validation():
    e = EvolutionEvent(kind="signal", payload={"n": 1}, ts=1.0)
    assert e.kind is EventKind.SIGNAL
    assert e.to_dict() == {"kind": "signal", "payload": {"n": 1}, "ts": 1.0}
    assert EvolutionEvent.from_dict(e.to_dict()) == e
    with pytest.raises(ValueError):
        EvolutionEvent(kind="not_a_kind")


def test_history_record_and_query(tmp_path):
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    history.emit(EventKind.SIGNAL, {"v": 1}, ts=10.0)
    history.emit(EventKind.REFLECT, {"v": 2}, ts=20.0)
    history.emit("signal", {"v": 3}, ts=30.0)  # 字符串 kind 也可

    all_events = history.query()
    assert [e.kind for e in all_events] == [
        EventKind.SIGNAL,
        EventKind.REFLECT,
        EventKind.SIGNAL,
    ]
    assert len(history.query(kind=EventKind.SIGNAL)) == 2
    assert len(history.query(kind="reflect")) == 1
    assert [e.ts for e in history.query(since=20.0)] == [20.0, 30.0]
    assert [e.ts for e in history.query(kind="signal", since=15.0)] == [30.0]


def test_history_empty_and_corrupt(tmp_path):
    history = EvolutionHistory(tmp_path / "empty.jsonl")
    assert history.query() == []
    assert history.metrics_series("x") == []
    # 损坏行必须报错而不是静默跳过
    history.path.write_text('{"kind": "signal"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        history.query()


def test_metrics_series(tmp_path):
    """指标序列供画 playbook 增长曲线 / 黄金集通过率曲线。"""
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    # 顶层键与 payload["metrics"] 嵌套两种口径都要能取到
    history.emit(EventKind.TRAIN_ROUND, {"playbook_size": 12}, ts=1.0)
    history.emit(EventKind.GATE_RESULT, {"metrics": {"gate.pass_rate": 0.9}}, ts=2.0)
    history.emit(EventKind.TRAIN_ROUND, {"playbook_size": 18}, ts=3.0)
    history.emit(EventKind.SIGNAL, {"playbook_size": "oops"}, ts=4.0)  # 非数值，跳过
    history.emit(EventKind.SIGNAL, {}, ts=5.0)  # 无该指标，跳过

    assert history.metrics_series("playbook_size") == [(1.0, 12.0), (3.0, 18.0)]
    assert history.metrics_series("gate.pass_rate") == [(2.0, 0.9)]
    assert history.metrics_series("nonexistent") == []


# ---------- 组合式插桩 ----------


def _golden() -> GoldenSet:
    return GoldenSet(
        [
            GoldenCase(input="重置密码", expected="设置"),
            GoldenCase(input="申请退款", expected="订单"),
        ]
    )


def _good(q: str) -> str:
    return {"重置密码": "请在设置里重置", "申请退款": "在订单页申请"}[q]


def _bad(q: str) -> str:
    return "不知道"


def test_wrap_gate_records_gate_result_and_promote(tmp_path):
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    gate = wrap_gate(EvolutionGate(_golden(), RuleBasedJudge()), history)
    result = gate.evaluate(_good)
    assert result.decision is GateDecision.PROMOTE

    kinds = [e.kind for e in history.query()]
    assert kinds == [EventKind.GATE_RESULT, EventKind.PROMOTE]
    gate_event = history.query(kind=EventKind.GATE_RESULT)[0]
    assert gate_event.payload["passed"] is True
    assert gate_event.payload["decision"] == "promote"
    # 指标序列可直接画通过率曲线
    assert [v for _, v in history.metrics_series("gate.pass_rate")] == [1.0]


def test_wrap_gate_records_rollback_with_baseline(tmp_path):
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    gate = wrap_gate(
        EvolutionGate(_golden(), RuleBasedJudge(), baseline=_good), history
    )
    result = gate.evaluate(_bad)
    assert result.decision is GateDecision.ROLLBACK
    assert [e.kind for e in history.query()] == [
        EventKind.GATE_RESULT,
        EventKind.ROLLBACK,
    ]


def test_wrap_gate_delegates_attributes(tmp_path):
    """组合包装：未拦截属性原样透传，包装对象可无缝替换。"""
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    inner = EvolutionGate(_golden(), RuleBasedJudge(), min_pass_rate=0.8)
    gate = wrap_gate(inner, history)
    assert gate.wrapped is inner
    assert gate.min_pass_rate == 0.8
    assert len(gate.golden_set) == 2
    assert gate.judge is inner.judge


def test_wrap_registry_records_promote_and_rollback(tmp_path):
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    registry = wrap_registry(AdapterRegistry(tmp_path / "reg"), history)
    passed = GateResult(passed=True, pass_rate=0.95, decision=GateDecision.PROMOTE)
    v1 = registry.register(tmp_path / "a1", metrics={})
    v2 = registry.register(tmp_path / "a2", metrics={})
    registry.promote(v1, passed)
    registry.promote(v2, passed)
    assert registry.current == v2
    assert registry.rollback() == v1

    promotes = history.query(kind=EventKind.PROMOTE)
    assert [e.payload["version"] for e in promotes] == [v1, v2]
    assert [v for _, v in history.metrics_series("gate.pass_rate")] == [0.95, 0.95]
    rollbacks = history.query(kind=EventKind.ROLLBACK)
    assert len(rollbacks) == 1
    assert rollbacks[0].payload["restored"] == v1


def test_wrap_registry_denied_promotion_records_halt(tmp_path):
    """红线语义：晋升被拒记录 HALT 事件，原异常继续抛出。"""
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    registry = wrap_registry(AdapterRegistry(tmp_path / "reg"), history)
    v = registry.register(tmp_path / "a1", metrics={})
    bad = GateResult(passed=False, pass_rate=0.1, decision=GateDecision.REJECT)
    with pytest.raises(PromotionDenied):
        registry.promote(v, bad)
    halts = history.query(kind=EventKind.HALT)
    assert len(halts) == 1
    assert halts[0].payload["reason"] == "promotion_denied"
    assert halts[0].payload["version"] == v
    assert registry.current is None


def test_wrap_registry_noop_rollback_records_nothing(tmp_path):
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    registry = wrap_registry(AdapterRegistry(tmp_path / "reg"), history)
    assert registry.rollback() is None
    assert history.query(kind=EventKind.ROLLBACK) == []


# ---------- 导出器（假模块模拟第三方依赖） ----------


def test_otel_exporter_missing_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # 强制 import 失败
    with pytest.raises(ImportError, match=r"evoflow\[otel\]"):
        OpenTelemetryExporter()


def test_langfuse_exporter_missing_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "langfuse", None)
    with pytest.raises(ImportError, match=r"evoflow\[otel\]"):
        LangfuseExporter()


class _FakeSpan:
    def __init__(self, name, sink):
        self.name = name
        self.sink = sink
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sink.append((self.name, self.attributes))
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self, sink):
        self.sink = sink

    def start_as_current_span(self, name):
        return _FakeSpan(name, self.sink)


def _install_fake_otel(monkeypatch, sink):
    fake = types.ModuleType("opentelemetry")
    fake_trace = types.SimpleNamespace(get_tracer=lambda name: _FakeTracer(sink))
    fake.trace = fake_trace
    monkeypatch.setitem(sys.modules, "opentelemetry", fake)


def test_otel_exporter_spans_and_incremental(monkeypatch, tmp_path):
    sink: list = []
    _install_fake_otel(monkeypatch, sink)
    exporter = OpenTelemetryExporter()
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    history.emit(EventKind.SIGNAL, {"score": 0.5, "meta": {"a": 1}}, ts=1.0)
    history.emit(EventKind.PROMOTE, {"version": "v0001"}, ts=2.0)

    assert exporter.export(history) == 2
    assert exporter.last_exported_ts == 2.0
    names = [name for name, _ in sink]
    assert names == ["evoflow.signal", "evoflow.promote"]
    attrs = sink[0][1]
    assert attrs["evoflow.event.kind"] == "signal"
    assert attrs["evoflow.score"] == 0.5
    assert attrs["evoflow.meta"] == '{"a": 1}'  # 非基本类型 JSON 化

    # 增量导出：只导新事件
    history.emit(EventKind.ROLLBACK, {"restored": "v0001"}, ts=3.0)
    assert exporter.export(history) == 1
    assert sink[-1][0] == "evoflow.rollback"
    assert exporter.export(history) == 0  # 无新事件


class _FakeLangfuseClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events = []

    def event(self, name, metadata=None):
        self.events.append((name, metadata))


def test_langfuse_exporter_events_and_incremental(monkeypatch, tmp_path):
    client = _FakeLangfuseClient()
    fake = types.ModuleType("langfuse")
    fake.Langfuse = lambda **kw: client
    monkeypatch.setitem(sys.modules, "langfuse", fake)

    exporter = LangfuseExporter()
    history = EvolutionHistory(tmp_path / "evo.jsonl")
    history.emit(EventKind.DISTILL, {"items": 3}, ts=1.0)
    assert exporter.export(history) == 1
    name, metadata = client.events[0]
    assert name == "evoflow.distill"
    assert metadata["items"] == 3
    assert metadata["ts"] == 1.0

    history.emit(EventKind.PAIR_BUILT, {"pairs": 7}, ts=2.0)
    assert exporter.export(history) == 1
    assert len(client.events) == 2
    assert exporter.export(history) == 0
