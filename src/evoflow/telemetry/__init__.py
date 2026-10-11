"""进化可观测性层：事件、历史流水与外部导出器。

自进化系统没有可观测性等于黑箱。本包提供：
- :class:`EvolutionEvent` / :class:`EventKind`：标准化进化事件。
- :class:`EvolutionHistory`：jsonl 追加式事件流水 + 查询 + 指标序列。
- :func:`wrap_gate` / :func:`wrap_registry`：组合式插桩（不修改原类）。
- :class:`OpenTelemetryExporter` / :class:`LangfuseExporter`：可选导出器
  （惰性依赖，需 ``pip install evoflow[otel]``）。
"""

from evoflow.telemetry.events import EventKind, EvolutionEvent
from evoflow.telemetry.exporters import LangfuseExporter, OpenTelemetryExporter
from evoflow.telemetry.history import EvolutionHistory, wrap_gate, wrap_registry

__all__ = [
    "EventKind",
    "EvolutionEvent",
    "EvolutionHistory",
    "LangfuseExporter",
    "OpenTelemetryExporter",
    "wrap_gate",
    "wrap_registry",
]
