"""EvoFlow MCP Server 测试（零第三方依赖，直接调 handler / 内存 StringIO）。

覆盖：
- 协议层：initialize / notifications / ping / tools/list / tools/call 全流程，
  未知方法 -32601、非法 JSON -32700、未知工具 -32602；
- 四个工具的行为与落盘副作用；
- 红线：tools/list 不含 promote/rollback/register/golden/judge 类工具，
  gate_status 不回传黄金集 failures 内容；
- 增强模式：mcp 包缺失时的清晰报错。
"""

import io
import json

import pytest

from evoflow.core.gate import GateResult
from evoflow.core.signals import SignalType
from evoflow.integrations import MCPServer, create_fastmcp_app, main
from evoflow.l0_context.playbook import Playbook
from evoflow.l2_weight.registry import AdapterRegistry


def make_server(tmp_path) -> MCPServer:
    return MCPServer(work_dir=tmp_path)


def rpc(server: MCPServer, method: str, params=None, msg_id=1) -> dict:
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle_message(msg)


def call_tool(server: MCPServer, name: str, arguments=None, msg_id=1) -> dict:
    """tools/call 并解析 content[0].text（JSON），返回 (isError, payload)。"""
    resp = rpc(server, "tools/call", {"name": name, "arguments": arguments or {}}, msg_id)
    assert "result" in resp, resp
    result = resp["result"]
    assert result["content"][0]["type"] == "text"
    text = result["content"][0]["text"]
    if result.get("isError"):
        return True, text
    return False, json.loads(text)


# ----------------------------------------------------------------------
# 协议层
# ----------------------------------------------------------------------


class TestProtocol:
    def test_initialize(self, tmp_path):
        server = make_server(tmp_path)
        resp = rpc(server, "initialize", {"protocolVersion": "2025-06-18"})
        result = resp["result"]
        assert resp["id"] == 1
        assert result["protocolVersion"] == "2025-06-18"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "evoflow"
        assert result["serverInfo"]["version"]

    def test_ping(self, tmp_path):
        server = make_server(tmp_path)
        assert rpc(server, "ping")["result"] == {}

    def test_initialized_notification_no_response(self, tmp_path):
        server = make_server(tmp_path)
        msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert server.handle_message(msg) is None

    def test_unknown_method_not_found(self, tmp_path):
        server = make_server(tmp_path)
        resp = rpc(server, "resources/list")
        assert resp["error"]["code"] == -32601
        assert "Method not found" in resp["error"]["message"]

    def test_unknown_lifecycle_method_not_found(self, tmp_path):
        """生命周期其他方法（如 prompts/list、completion/complete）返回 method not found。"""
        server = make_server(tmp_path)
        for method in ("prompts/list", "resources/read", "completion/complete", "shutdown"):
            resp = rpc(server, method)
            assert resp["error"]["code"] == -32601, method

    def test_parse_error_on_invalid_json(self, tmp_path):
        server = make_server(tmp_path)
        out = server.handle_line("{not valid json")
        resp = json.loads(out)
        assert resp["error"]["code"] == -32700
        assert resp["id"] is None

    def test_invalid_request_missing_jsonrpc(self, tmp_path):
        server = make_server(tmp_path)
        resp = server.handle_message({"id": 9, "method": "ping"})
        assert resp["error"]["code"] == -32600

    def test_stdio_full_flow_over_stringio(self, tmp_path):
        """内存 stdin/stdout 全流程：initialize → 通知（跳过）→ tools/list → tools/call。"""
        server = make_server(tmp_path)
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "evoflow_gate_status", "arguments": {}},
                }
            ),
            "",  # 空行应被跳过
        ]
        instream = io.StringIO("\n".join(lines) + "\n")
        outstream = io.StringIO()
        server.serve(instream, outstream)

        outputs = [
            json.loads(line) for line in outstream.getvalue().splitlines() if line.strip()
        ]
        assert len(outputs) == 3  # 通知无响应
        assert outputs[0]["id"] == 1 and "serverInfo" in outputs[0]["result"]
        assert outputs[1]["id"] == 2 and "tools" in outputs[1]["result"]
        assert outputs[2]["id"] == 3 and outputs[2]["result"]["isError"] is False

    def test_main_entry_with_eof_stdin(self, tmp_path, monkeypatch):
        """python -m 入口：EOF 即正常退出（返回 0）。"""
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert main(["--work-dir", str(tmp_path)]) == 0


# ----------------------------------------------------------------------
# tools/list 与红线
# ----------------------------------------------------------------------


class TestToolList:
    EXPECTED = {
        "evoflow_recall",
        "evoflow_feedback",
        "evoflow_reflect",
        "evoflow_gate_status",
    }

    def _names(self, tmp_path):
        server = make_server(tmp_path)
        resp = rpc(server, "tools/list")
        return [t["name"] for t in resp["result"]["tools"]]

    def test_exactly_four_tools(self, tmp_path):
        assert set(self._names(tmp_path)) == self.EXPECTED

    def test_schemas_have_required_fields(self, tmp_path):
        server = make_server(tmp_path)
        tools = rpc(server, "tools/list")["result"]["tools"]
        for tool in tools:
            assert tool["description"]
            assert tool["inputSchema"]["type"] == "object"
        schemas = {t["name"]: t["inputSchema"] for t in tools}
        assert schemas["evoflow_recall"]["required"] == ["query"]
        assert set(schemas["evoflow_feedback"]["required"]) == {
            "task_id",
            "signal_type",
            "value",
        }
        assert set(schemas["evoflow_reflect"]["required"]) == {"task", "steps", "success"}
        # signal_type 枚举与 SignalType 一致
        enum = schemas["evoflow_feedback"]["properties"]["signal_type"]["enum"]
        assert set(enum) == {s.value for s in SignalType}

    def test_red_line_no_promote_or_eval_tools(self, tmp_path):
        """红线：不开放晋升/回滚/注册/训练，不暴露黄金集与 judge。"""
        forbidden = ("promote", "rollback", "register", "train", "golden", "judge", "gate_run")
        for name in self._names(tmp_path):
            for word in forbidden:
                assert word not in name.lower(), f"红线违规工具: {name}"

    def test_unknown_tool_invalid_params(self, tmp_path):
        server = make_server(tmp_path)
        resp = rpc(server, "tools/call", {"name": "evoflow_promote", "arguments": {}})
        assert resp["error"]["code"] == -32602
        assert "未知工具" in resp["error"]["message"]


# ----------------------------------------------------------------------
# 四个工具的行为
# ----------------------------------------------------------------------


class TestRecall:
    def test_recall_empty_bank(self, tmp_path):
        server = make_server(tmp_path)
        is_err, payload = call_tool(server, "evoflow_recall", {"query": "如何退款"})
        assert not is_err
        assert payload["query"] == "如何退款"
        assert payload["strategies"] == []
        assert payload["playbook_entries"] == 0
        assert "Playbook" in payload["playbook"]

    def test_recall_after_reflect_contains_entry(self, tmp_path):
        server = make_server(tmp_path)
        call_tool(
            server,
            "evoflow_reflect",
            {"task": "处理客户退款", "steps": ["查订单", "原路退回"], "success": True},
        )
        is_err, payload = call_tool(server, "evoflow_recall", {"query": "退款", "k": 1})
        assert not is_err
        assert "处理客户退款" in payload["playbook"]
        assert payload["playbook_entries"] == 1


class TestFeedback:
    def test_feedback_recorded_and_persisted(self, tmp_path):
        server = make_server(tmp_path)
        is_err, payload = call_tool(
            server,
            "evoflow_feedback",
            {
                "task_id": "t-001",
                "signal_type": "correction",
                "value": 1.0,
                "meta": {"original": "错", "corrected": "对"},
            },
        )
        assert not is_err
        assert payload["recorded"] is True
        assert payload["signal"]["type"] == "correction"
        assert payload["total_signals"] == 1
        # 落盘：新进程可通过 SignalCollector 恢复
        signals_file = tmp_path / "data" / "signals.jsonl"
        assert signals_file.exists()
        row = json.loads(signals_file.read_text(encoding="utf-8").strip())
        assert row["task_id"] == "t-001"
        assert row["meta"]["corrected"] == "对"

    def test_feedback_invalid_signal_type_is_tool_error(self, tmp_path):
        server = make_server(tmp_path)
        resp = rpc(
            server,
            "tools/call",
            {
                "name": "evoflow_feedback",
                "arguments": {"task_id": "t", "signal_type": "maybe", "value": 1},
            },
        )
        result = resp["result"]
        assert result["isError"] is True
        assert "signal_type" in result["content"][0]["text"]

    def test_feedback_all_signal_types_accepted(self, tmp_path):
        server = make_server(tmp_path)
        for i, st in enumerate(SignalType):
            is_err, payload = call_tool(
                server,
                "evoflow_feedback",
                {"task_id": f"t-{i}", "signal_type": st.value, "value": 0.5},
            )
            assert not is_err, st
        assert payload["total_signals"] == len(SignalType)


class TestReflect:
    def test_reflect_success_curates_and_persists(self, tmp_path):
        server = make_server(tmp_path)
        is_err, payload = call_tool(
            server,
            "evoflow_reflect",
            {
                "task": "修复登录 Bug",
                "steps": ["复现", "定位为 token 过期", "刷新逻辑修复"],
                "success": True,
                "detail": "单测通过",
            },
        )
        assert not is_err
        assert payload["added"] == 1
        assert payload["playbook_entries"] == 1
        assert payload["entries"][0]["tags"] == ["reflect", "success"]
        # 落盘后可重新加载
        reloaded = Playbook.load(tmp_path / "playbook.json")
        assert len(reloaded) == 1
        assert "修复登录 Bug" in reloaded.entries[0].content

    def test_reflect_fail_entry_tagged_fail(self, tmp_path):
        server = make_server(tmp_path)
        is_err, payload = call_tool(
            server,
            "evoflow_reflect",
            {
                "task": "发布版本",
                "steps": ["直接推 main"],
                "success": False,
                "detail": "CI 红",
            },
        )
        assert not is_err
        assert payload["entries"][0]["tags"] == ["reflect", "fail"]
        assert "CI 红" in payload["entries"][0]["content"]

    def test_reflect_dedup_same_trajectory(self, tmp_path):
        """同内容轨迹反思两次，Playbook 去重不膨胀。"""
        server = make_server(tmp_path)
        args = {"task": "同上", "steps": ["步骤A"], "success": True}
        call_tool(server, "evoflow_reflect", args)
        is_err, payload = call_tool(server, "evoflow_reflect", args)
        assert not is_err
        assert payload["added"] == 0
        assert payload["playbook_entries"] == 1


class TestGateStatus:
    def test_empty_registry_status(self, tmp_path):
        server = make_server(tmp_path)
        is_err, payload = call_tool(server, "evoflow_gate_status")
        assert not is_err
        assert payload["current_adapter"] is None
        assert payload["versions"] == {}
        assert payload["promotion_history"] == []
        assert payload["playbook_entries"] == 0
        assert payload["bank_size"] == 0
        assert payload["signals"] == 0
        assert payload["readonly"] is True

    def test_status_after_operations(self, tmp_path):
        server = make_server(tmp_path)
        call_tool(server, "evoflow_reflect", {"task": "T", "steps": ["s"], "success": True})
        call_tool(
            server,
            "evoflow_feedback",
            {"task_id": "T", "signal_type": "success", "value": 1.0},
        )
        is_err, payload = call_tool(server, "evoflow_gate_status")
        assert not is_err
        assert payload["playbook_entries"] == 1
        assert payload["signals"] == 1

    def test_red_line_no_golden_set_leak(self, tmp_path):
        """红线：注册表中的 gate failures（含黄金集内容）不得经 MCP 外泄。"""
        registry = AdapterRegistry(tmp_path / "adapters")
        registry.register(
            "adapters/v0001",
            metrics={"dpo_loss": 0.1},
            gate_result=GateResult(
                passed=True,
                pass_rate=0.95,
                failures=[{"input": "SECRET-GOLDEN-CASE", "expected": "x", "output": "y"}],
            ),
        )
        registry.promote("v0001")
        server = make_server(tmp_path)
        is_err, payload = call_tool(server, "evoflow_gate_status")
        assert not is_err
        assert payload["current_adapter"] == "v0001"
        assert payload["versions"]["v0001"]["gate_passed"] is True
        assert payload["versions"]["v0001"]["gate_pass_rate"] == 0.95
        blob = json.dumps(payload, ensure_ascii=False)
        assert "SECRET-GOLDEN-CASE" not in blob
        assert "failures" not in blob


# ----------------------------------------------------------------------
# 增强模式（FastMCP）
# ----------------------------------------------------------------------


class TestFastMCP:
    def test_missing_sdk_raises_clear_error(self, tmp_path):
        try:
            import mcp  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match=r"pip install evoflow\[mcp\]"):
                create_fastmcp_app(work_dir=tmp_path)
        else:
            pytest.skip("已安装 mcp 包，走 test_app_built_when_sdk_available")

    def test_app_built_when_sdk_available(self, tmp_path):
        pytest.importorskip("mcp")
        app = create_fastmcp_app(work_dir=tmp_path)
        assert app.name == "evoflow"


# ----------------------------------------------------------------------
# 导出契约
# ----------------------------------------------------------------------


def test_integrations_exports():
    from evoflow import integrations

    assert integrations.MCPServer is MCPServer
    assert integrations.create_fastmcp_app is create_fastmcp_app
    assert callable(integrations.main)
