"""可选导出器：把进化事件流水导出到外部可观测性平台。

两者都**惰性 import** 第三方依赖——核心包保持零第三方依赖的硬承诺；
依赖缺失时在构造阶段抛出清晰错误（``请 pip install evoflow[otel]``）。

都以 :meth:`EvolutionHistory.query` 为输入，并记录已导出的最大 ts，
重复调用 :meth:`export` 只做**增量导出**，不产生重复 span / event。
"""

from __future__ import annotations

import json

from evoflow.telemetry.events import EventKind, EvolutionEvent
from evoflow.telemetry.history import EvolutionHistory

_OTEL_HINT = (
    "缺少 opentelemetry 依赖，无法使用 OpenTelemetryExporter；"
    "请 pip install evoflow[otel]"
)
_LANGFUSE_HINT = (
    "缺少 langfuse 依赖，无法使用 LangfuseExporter；"
    "请 pip install evoflow[otel]"
)


def _kind_value(event: EvolutionEvent) -> str:
    """事件类别字符串（EvolutionEvent 构造时已校正为 EventKind，防御式处理 str）。"""
    return event.kind.value if isinstance(event.kind, EventKind) else event.kind


def _span_attributes(event: EvolutionEvent) -> dict:
    """把事件载荷压平为 OTel span attributes（值必须是基本类型）。"""
    attrs = {"evoflow.event.kind": _kind_value(event), "evoflow.event.ts": event.ts}
    for key, value in event.payload.items():
        attr_key = f"evoflow.{key}"
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            attrs[attr_key] = value
        else:
            attrs[attr_key] = json.dumps(value, ensure_ascii=False, default=str)
    return attrs


class OpenTelemetryExporter:
    """把进化事件转成 OTel span 的导出器。

    :param service_name: tracer 名称（缺省 ``"evoflow"``）。
    :param tracer: 可选注入的 tracer（测试/自定义管线用）；缺省通过
        ``opentelemetry.trace.get_tracer(service_name)`` 获取。
    """

    def __init__(self, service_name: str = "evoflow", tracer=None) -> None:
        try:
            from opentelemetry import trace
        except ImportError as e:
            raise ImportError(_OTEL_HINT) from e
        self._tracer = tracer if tracer is not None else trace.get_tracer(service_name)
        self._last_ts = 0.0

    @property
    def last_exported_ts(self) -> float:
        """已导出事件的最大 ts（增量水位线）。"""
        return self._last_ts

    def export(self, history: EvolutionHistory) -> int:
        """增量导出 history 中 ts 大于水位线的事件，返回导出条数。"""
        events = [e for e in history.query() if e.ts > self._last_ts]
        for event in events:
            with self._tracer.start_as_current_span(
                f"evoflow.{_kind_value(event)}"
            ) as span:
                for key, value in _span_attributes(event).items():
                    span.set_attribute(key, value)
        if events:
            self._last_ts = max(e.ts for e in events)
        return len(events)


class LangfuseExporter:
    """把进化事件导出到 Langfuse 的导出器。

    :param client: 可选注入的 Langfuse 客户端（测试/自定义配置用）；
        缺省以 ``langfuse.Langfuse(**client_kwargs)`` 构造。
    :param client_kwargs: 传给 Langfuse 客户端的参数（如 host / 密钥）。
    """

    def __init__(self, client=None, **client_kwargs) -> None:
        try:
            import langfuse
        except ImportError as e:
            raise ImportError(_LANGFUSE_HINT) from e
        self._client = client if client is not None else langfuse.Langfuse(**client_kwargs)
        self._last_ts = 0.0

    @property
    def last_exported_ts(self) -> float:
        """已导出事件的最大 ts（增量水位线）。"""
        return self._last_ts

    def export(self, history: EvolutionHistory) -> int:
        """增量导出 history 中 ts 大于水位线的事件，返回导出条数。"""
        events = [e for e in history.query() if e.ts > self._last_ts]
        for event in events:
            self._client.event(
                name=f"evoflow.{_kind_value(event)}",
                metadata={"ts": event.ts, **event.payload},
            )
        if events:
            self._last_ts = max(e.ts for e in events)
        return len(events)
