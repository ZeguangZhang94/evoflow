"""EvoFlow MCP Server 零依赖演示。

不起真实 stdio 服务，而是直接实例化 :class:`MCPServer`，逐条喂 JSON-RPC
消息（与 stdio 模式下宿主发送的消息完全一致），打印每步响应要点：

1. ``initialize``           —— 协议握手，返回 serverInfo 与 instructions；
2. ``tools/list``           —— 列出全部工具；
3. ``tools/call`` 四次      —— evoflow_recall / evoflow_feedback /
   evoflow_reflect / evoflow_gate_status 各调一次。

红线演示：工具列表中**没有 promote**——晋升/回滚不通过 MCP 开放，
防止运行中的 agent 自己给自己晋升。

运行：``python examples/mcp_demo.py``（零依赖）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 未安装包时直接从 src 布局导入，保证裸环境可跑。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoflow.integrations.mcp_server import MCPServer

WORK_DIR = Path(__file__).resolve().parent / "_demo_out" / "mcp"


def hr(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def rpc(server: MCPServer, msg_id: int, method: str, params: dict | None = None) -> dict:
    """发送一条 JSON-RPC 请求并返回响应（即 stdio 模式下的一问一答）。"""
    message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    response = server.handle_message(message)
    assert response is not None, f"{method} 应有响应"
    return response


def main() -> None:
    server = MCPServer(work_dir=WORK_DIR)
    print(f"MCP Server 已实例化（work_dir={WORK_DIR}，纯标准库 JSON-RPC 模式）")

    # ---------------------------------------------------------- 1. initialize
    hr("1. initialize：协议握手")
    resp = rpc(server, 1, "initialize")
    result = resp["result"]
    print(f"  protocolVersion: {result['protocolVersion']}")
    print(f"  serverInfo     : {result['serverInfo']}")
    print(f"  instructions   : {result['instructions'][:60]}...")

    # ---------------------------------------------------------- 2. tools/list
    hr("2. tools/list：工具清单")
    resp = rpc(server, 2, "tools/list")
    tools = resp["result"]["tools"]
    names = [t["name"] for t in tools]
    for t in tools:
        print(f"  - {t['name']}: {t['description'][:40]}...")
    print(f"\n  红线检查：工具列表共 {len(tools)} 个 -> {names}")
    assert not any("promote" in n for n in names), "工具列表不应出现 promote"
    print("  ✔ 无 promote 类工具：晋升/回滚不通过 MCP 开放（红线生效）")

    # ---------------------------------------------------------- 3. tools/call
    hr("3. tools/call：依次调用四个工具")

    def call(msg_id: int, name: str, arguments: dict) -> dict:
        resp = rpc(server, msg_id, "tools/call", {"name": name, "arguments": arguments})
        result = resp["result"]
        assert not result.get("isError"), f"{name} 执行失败: {result}"
        # MCP 约定：工具结果在 content[0].text（JSON 字符串）
        return json.loads(result["content"][0]["text"])

    out = call(3, "evoflow_recall", {"query": "退款多久到账", "k": 1})
    print(f"  evoflow_recall    : playbook_entries={out['playbook_entries']}, "
          f"strategies={len(out['strategies'])}")

    out = call(4, "evoflow_feedback", {
        "task_id": "退款进度咨询",
        "signal_type": "success",
        "value": 1.0,
    })
    print(f"  evoflow_feedback  : recorded={out['recorded']}, "
          f"total_signals={out['total_signals']}（已追加落盘）")

    out = call(5, "evoflow_reflect", {
        "task": "退款进度咨询",
        "steps": ["查询订单状态", "回复：退款 3 个工作日内原路退回"],
        "success": True,
        "detail": "用户满意",
    })
    print(f"  evoflow_reflect   : added={out['added']} 条 delta, "
          f"playbook_entries={out['playbook_entries']}（已落盘 playbook.json）")

    out = call(6, "evoflow_gate_status", {})
    print(f"  evoflow_gate_status: current_adapter={out['current_adapter']}, "
          f"readonly={out['readonly']}")
    print(f"    note: {out['note']}")
    assert "failures" not in str(out), "gate_status 不应泄露黄金集失败用例"
    print("  ✔ gate_status 只读且不含黄金集内容（评分器隔离红线生效）")

    hr("mcp demo 完成")
    print(f"产物目录: {WORK_DIR}（signals.jsonl / playbook.json）")


if __name__ == "__main__":
    main()
