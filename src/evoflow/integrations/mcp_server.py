"""EvoFlow MCP Server —— 把进化能力包装为 MCP 工具，供 Claude Code / Cursor 等宿主调用。

双模式：

1. **纯标准库模式（默认）**：:class:`MCPServer` 实现 JSON-RPC 2.0 over stdio
   的最小 MCP 协议子集（``initialize`` / ``ping`` / ``tools/list`` / ``tools/call``，
   协议版本 2025-06-18 的 tools 能力部分），不依赖 mcp 包，保住核心包零依赖承诺。
   生命周期其他方法一律返回 ``Method not found``（-32601）。
2. **增强模式**：:func:`create_fastmcp_app` 惰性 import 官方 mcp 包
   （FastMCP），未安装时抛出带安装指引的 ImportError
   （``pip install evoflow[mcp]``）。暴露的工具集与标准库模式完全一致。

工具集（均为 EvoFlow 现有 API 的薄封装）：

- ``evoflow_recall``     ：ReasoningBank 检索 + Playbook.render 合并返回（执行前调用）。
- ``evoflow_feedback``   ：上报反馈信号（accept/reject/success/fail/correction）。
- ``evoflow_reflect``    ：对一条轨迹反思，把 delta 策展进 Playbook 并落盘。
- ``evoflow_gate_status``：只读状态（adapter 当前版本、注册表摘要、条目计数）。

红线（与评估侧隔离，同 docs/integration.md §0）：

- **不暴露** golden set 内容与 judge 细节；gate_status 只回传 gate 的
  passed / pass_rate 摘要，不回传 failures（其中含黄金集 input/expected）。
- gate_status **只读**；晋升 / 回滚 / 注册均不通过 MCP 开放——
  防止运行中的 agent 自己给自己晋升。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Any

from evoflow import __version__
from evoflow.core.signals import SignalCollector, SignalType
from evoflow.core.trajectory import Outcome, Trajectory
from evoflow.l0_context.playbook import Playbook
from evoflow.l0_context.reflector import Reflector
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore
from evoflow.l2_weight.registry import AdapterRegistry

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "evoflow"

# JSON-RPC 2.0 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

MCP_INSTALL_HINT = "MCP 增强模式需要官方 SDK，请 pip install evoflow[mcp]"


class _ToolNotFound(Exception):
    """tools/call 请求了未注册的工具。"""


def _result(msg_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _default_reflect_fn(trajectory: Trajectory) -> list[str]:
    """零依赖规则反思器：成功提炼做法、失败提炼避坑（与 ReasoningBank 默认蒸馏器同风格）。"""
    steps_text = " -> ".join(trajectory.steps[:5]) or "（无步骤记录）"
    task_short = trajectory.task[:30]
    if trajectory.outcome.success:
        return [f"做法：任务「{task_short}」可按步骤 [{steps_text}] 复现成功"]
    detail = trajectory.outcome.detail or "原因未明"
    return [f"避坑：任务「{task_short}」失败（{detail}），勿复用步骤 [{steps_text}]"]


class MCPServer:
    """EvoFlow MCP Server（纯标准库，JSON-RPC 2.0 over stdio）。

    服务器持有 Playbook / ReasoningBank / AdapterRegistry / SignalCollector
    实例，按工作目录约定布局读写::

        <work_dir>/
          playbook.json             # L0 playbook（reflect 后自动落盘）
          data/signals.jsonl        # 反馈信号（追加写）
          data/reasoning_bank.jsonl # L1 记忆库
          adapters/                 # L2 adapter 注册表（只读）

    :param work_dir: EvoFlow 工作目录（与 ``evoflow init`` 生成的骨架一致）。
    """

    def __init__(self, work_dir: str | Path = ".") -> None:
        self.work_dir = Path(work_dir)
        data_dir = self.work_dir / "data"
        self.playbook_path = self.work_dir / "playbook.json"
        self.bank_path = data_dir / "reasoning_bank.jsonl"

        self.playbook = Playbook.load(self.playbook_path)
        store = JsonlVectorStore()
        store.load(self.bank_path)
        self.bank = ReasoningBank(store)
        self.collector = SignalCollector(data_dir / "signals.jsonl")
        self.registry = AdapterRegistry(self.work_dir / "adapters")
        self.reflector = Reflector(_default_reflect_fn)

    # ------------------------------------------------------------------
    # 工具实现（薄封装，返回可 JSON 序列化的 dict）
    # ------------------------------------------------------------------

    def tool_recall(self, query: str, k: int = 1) -> dict:
        """检索经验：ReasoningBank 检索 + Playbook 渲染，合并返回（执行前调用）。"""
        k = max(1, int(k))
        strategies = self.bank.retrieve(query, k=k)
        return {
            "query": query,
            "playbook": self.playbook.render(),
            "playbook_entries": len(self.playbook),
            "strategies": [s.to_dict() for s in strategies],
        }

    def tool_feedback(
        self,
        task_id: str,
        signal_type: str,
        value: float,
        meta: dict | None = None,
    ) -> dict:
        """上报一条反馈信号（立即追加落盘，崩溃不丢）。"""
        try:
            st = SignalType(signal_type)
        except ValueError:
            allowed = [s.value for s in SignalType]
            raise ValueError(f"非法 signal_type: {signal_type!r}，允许值: {allowed}") from None
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("meta 必须是 object")
        signal = self.collector.emit(st, task_id=task_id, value=float(value), meta=meta or {})
        return {
            "recorded": True,
            "signal": signal.to_dict(),
            "total_signals": len(self.collector),
        }

    def tool_reflect(
        self,
        task: str,
        steps: list[str],
        success: bool,
        detail: str = "",
    ) -> dict:
        """对一条轨迹反思，把 delta 策展进 Playbook 并落盘。"""
        trajectory = Trajectory(
            task=task,
            steps=[str(s) for s in steps],
            outcome=Outcome(success=bool(success), verifier="mcp-host", detail=detail or ""),
        )
        deltas = self.reflector.reflect(trajectory)
        added = self.playbook.curate(deltas)
        self.playbook.save(self.playbook_path)
        return {
            "added": len(added),
            "entries": [e.to_dict() for e in added],
            "playbook_entries": len(self.playbook),
            "playbook_path": str(self.playbook_path),
        }

    def tool_gate_status(self) -> dict:
        """只读状态摘要。

        红线：只回传 gate 的 passed / pass_rate 摘要，**不**回传 failures
        （含黄金集内容）；晋升 / 回滚不通过 MCP 开放。
        """
        versions: dict[str, dict] = {}
        for v in self.registry.versions():
            info = self.registry.info(v)
            gate = info.get("gate") or {}
            versions[v] = {
                "registered_at": info.get("registered_at"),
                "gate_passed": gate.get("passed"),
                "gate_pass_rate": gate.get("pass_rate"),
            }
        history: list[str] = []
        index_path = self.registry.root / AdapterRegistry.INDEX_FILE
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                history = list(json.load(f).get("history", []))
        return {
            "current_adapter": self.registry.current,
            "versions": versions,
            "promotion_history": history,
            "playbook_entries": len(self.playbook),
            "bank_size": len(self.bank),
            "signals": len(self.collector),
            "work_dir": str(self.work_dir.resolve()),
            "readonly": True,
            "note": "晋升/回滚不通过 MCP 开放（红线：防止 agent 自行晋升）",
        }

    # ------------------------------------------------------------------
    # 工具元数据
    # ------------------------------------------------------------------

    def _tool_specs(self) -> list[dict]:
        """tools/list 返回的工具描述（name / description / inputSchema）。"""
        return [
            {
                "name": "evoflow_recall",
                "description": (
                    "检索 EvoFlow 经验库：ReasoningBank 策略检索 + Playbook 渲染，"
                    "合并返回可注入 prompt 的上下文。agent 执行任务前调用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "检索查询（通常为当前任务描述）",
                        },
                        "k": {
                            "type": "integer",
                            "default": 1,
                            "minimum": 1,
                            "description": "返回策略条数，默认 1（原则：相关性 > 数量）",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "evoflow_feedback",
                "description": (
                    "上报一条反馈信号（accept/reject/success/fail/correction），"
                    "作为三层进化栈的原始燃料，立即落盘。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "关联的任务标识"},
                        "signal_type": {
                            "type": "string",
                            "enum": [s.value for s in SignalType],
                            "description": "信号类型",
                        },
                        "value": {
                            "type": "number",
                            "description": "数值化强度（如 +1/-1 或 0~1 评分）",
                        },
                        "meta": {
                            "type": "object",
                            "default": {},
                            "description": "附加信息；correction 信号约定含 original/corrected 键",
                        },
                    },
                    "required": ["task_id", "signal_type", "value"],
                },
            },
            {
                "name": "evoflow_reflect",
                "description": (
                    "对一条任务轨迹做反思，把增量经验条目（delta）策展进 Playbook 并落盘。"
                    "任务结束后调用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "任务描述"},
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "执行步骤序列（最后一步视为最终回答）",
                        },
                        "success": {"type": "boolean", "description": "任务是否成功"},
                        "detail": {
                            "type": "string",
                            "default": "",
                            "description": "判定细节（失败原因、评分依据等）",
                        },
                    },
                    "required": ["task", "steps", "success"],
                },
            },
            {
                "name": "evoflow_gate_status",
                "description": (
                    "查询只读状态：当前 adapter 版本、注册表历史摘要、playbook/记忆库/"
                    "信号计数。不含黄金集内容与 judge 细节；晋升/回滚不通过 MCP 开放。"
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def _tool_handlers(self) -> dict:
        return {
            "evoflow_recall": self.tool_recall,
            "evoflow_feedback": self.tool_feedback,
            "evoflow_reflect": self.tool_reflect,
            "evoflow_gate_status": self.tool_gate_status,
        }

    # ------------------------------------------------------------------
    # JSON-RPC 2.0 协议处理
    # ------------------------------------------------------------------

    def _handle_initialize(self) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            "instructions": (
                "EvoFlow 进化能力：执行前调 evoflow_recall 取经验；执行后调 "
                "evoflow_feedback 上报信号、evoflow_reflect 沉淀经验；"
                "evoflow_gate_status 提供只读状态。"
            ),
        }

    def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _ToolNotFound(f"tools/call 的 arguments 必须是 object: {name!r}")
        handler = self._tool_handlers().get(name)
        if handler is None:
            raise _ToolNotFound(f"未知工具: {name!r}")
        try:
            output = handler(**arguments)
        except (ValueError, KeyError, TypeError) as e:
            # 工具执行错误：按 MCP 约定放进 result.isError，而非 JSON-RPC 错误
            return {
                "content": [{"type": "text", "text": f"工具执行失败: {e}"}],
                "isError": True,
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(output, ensure_ascii=False, indent=2),
                }
            ],
            "isError": False,
        }

    def handle_message(self, message: Any) -> dict | None:
        """处理一条 JSON-RPC 消息，返回响应 dict；通知（无 id）返回 None。

        :param message: 已反序列化的 JSON-RPC 消息。
        """
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            msg_id = message.get("id") if isinstance(message, dict) else None
            return _error(msg_id, INVALID_REQUEST, "Invalid Request")

        method = message["method"]
        msg_id = message.get("id")
        is_notification = "id" not in message

        # 通知（notifications/*）无需响应
        if method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result: Any = self._handle_initialize()
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self._tool_specs()}
            elif method == "tools/call":
                result = self._handle_tools_call(message.get("params") or {})
            else:
                if is_notification:
                    return None
                return _error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except _ToolNotFound as e:
            if is_notification:
                return None
            return _error(msg_id, INVALID_PARAMS, str(e))
        except Exception as e:  # noqa: BLE001 —— 协议层兜底，绝不让异常杀死 stdio 循环
            if is_notification:
                return None
            return _error(msg_id, INTERNAL_ERROR, f"Internal error: {e}")

        if is_notification:
            return None
        return _result(msg_id, result)

    def handle_line(self, line: str) -> str | None:
        """处理一行 stdio 输入（NDJSON），返回响应行；通知/无需响应时返回 None。"""
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return json.dumps(_error(None, PARSE_ERROR, "Parse error"), ensure_ascii=False)
        response = self.handle_message(message)
        if response is None:
            return None
        return json.dumps(response, ensure_ascii=False)

    def serve(
        self,
        instream: IO[str] | None = None,
        outstream: IO[str] | None = None,
    ) -> None:
        """stdio 服务循环：逐行读 JSON-RPC 消息，逐行写响应（NDJSON，无其他输出）。"""
        instream = instream if instream is not None else sys.stdin
        outstream = outstream if outstream is not None else sys.stdout
        for line in instream:
            line = line.strip()
            if not line:
                continue
            out = self.handle_line(line)
            if out is not None:
                outstream.write(out + "\n")
                outstream.flush()


def create_fastmcp_app(work_dir: str | Path = "."):
    """增强模式：用官方 MCP SDK（FastMCP）暴露同样的四个工具。

    惰性 import mcp 包；未安装时抛 ImportError（含 ``pip install evoflow[mcp]`` 指引）。

    :param work_dir: EvoFlow 工作目录，语义同 :class:`MCPServer`。
    :return: ``mcp.server.fastmcp.FastMCP`` 实例（``app.run()`` 即可启动 stdio 服务）。
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(MCP_INSTALL_HINT) from e

    server = MCPServer(work_dir=work_dir)
    app = FastMCP(name=SERVER_NAME)

    @app.tool(
        name="evoflow_recall",
        description="检索 EvoFlow 经验库（ReasoningBank + Playbook），agent 执行任务前调用。",
    )
    def _recall(query: str, k: int = 1) -> dict:
        return server.tool_recall(query, k)

    @app.tool(
        name="evoflow_feedback",
        description="上报反馈信号（accept/reject/success/fail/correction），立即落盘。",
    )
    def _feedback(task_id: str, signal_type: str, value: float, meta: dict | None = None) -> dict:
        return server.tool_feedback(task_id, signal_type, value, meta)

    @app.tool(
        name="evoflow_reflect",
        description="对任务轨迹反思，把增量经验策展进 Playbook 并落盘。",
    )
    def _reflect(task: str, steps: list[str], success: bool, detail: str = "") -> dict:
        return server.tool_reflect(task, steps, success, detail)

    @app.tool(
        name="evoflow_gate_status",
        description="只读状态：adapter 当前版本、注册表摘要、条目计数（晋升/回滚不开放）。",
    )
    def _gate_status() -> dict:
        return server.tool_gate_status()

    return app


def main(argv: list[str] | None = None) -> int:
    """stdio 服务入口：``python -m evoflow.integrations.mcp_server [--work-dir DIR]``。"""
    parser = argparse.ArgumentParser(
        prog="evoflow-mcp",
        description="EvoFlow MCP Server（JSON-RPC 2.0 over stdio，纯标准库）",
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="EvoFlow 工作目录（含 playbook.json / data/ / adapters/，默认当前目录）",
    )
    args = parser.parse_args(argv)
    MCPServer(work_dir=args.work_dir).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
