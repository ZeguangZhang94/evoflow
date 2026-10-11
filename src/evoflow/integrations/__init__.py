"""EvoFlow 外部生态集成（MCP Server 等）。

本目录为可选集成层，不破坏核心包的零第三方依赖承诺：
:mod:`evoflow.integrations.mcp_server` 的默认模式（:class:`MCPServer`）
为纯标准库实现；增强模式（:func:`create_fastmcp_app`）惰性 import 官方
mcp 包，未安装时抛出含 ``pip install evoflow[mcp]`` 指引的 ImportError。

启动 stdio 服务::

    python -m evoflow.integrations.mcp_server --work-dir /path/to/project

导出采用 PEP 562 惰性加载，避免 ``python -m`` 启动时的重复导入告警。
"""

from typing import TYPE_CHECKING

__all__ = ["MCPServer", "create_fastmcp_app", "main"]

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from evoflow.integrations.mcp_server import MCPServer, create_fastmcp_app, main


def __getattr__(name: str):
    if name in __all__:
        from evoflow.integrations import mcp_server

        return getattr(mcp_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
