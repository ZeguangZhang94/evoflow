# 安全政策 / Security Policy

## 支持的版本 / Supported Versions

| 版本 Version | 支持状态 Supported |
|---|---|
| 0.2.x | ✅ 当前版本，接受安全修复 |
| 0.1.x | ❌ 请升级到 0.2.x |

## 报告漏洞 / Reporting a Vulnerability

**请勿通过公开 Issue 报告安全漏洞。**
**Please do NOT report security vulnerabilities through public GitHub issues.**

请通过以下方式私下报告 / Please report privately via：

- GitHub 私密漏洞报告（推荐）：仓库页面的 **Security → Report a vulnerability**
- 或直接在 Issue 中 @维护者 要求私下联系，**不要附漏洞细节**

请在报告中包含 / Please include：

- 受影响的版本与模块
- 复现步骤或概念验证（PoC）
- 潜在影响评估

## 响应承诺 / Response Commitment

- **48 小时内**确认收到报告
- **7 天内**给出初步评估与修复计划
- 修复发布后，经报告者同意会在 CHANGELOG 与 Release Notes 中致谢

## 安全相关设计说明 / Security-relevant Design Notes

本项目包含若干"安全红线"设计，与安全审计相关：

1. **评分器隔离**：黄金集与 judge 细节对 agent 运行时不可见（防 reward hacking）——见 `evoflow/core/gate.py` 与 `evoflow/eval/judge.py`。
2. **评估门硬约束**：未过线禁止晋升，回滚一键可用——见 `evoflow/l2_weight/registry.py`。
3. **黄金集防泄漏**：训练启动前的泄漏检测——见 `evoflow/eval/leakage.py`。
4. **MCP 工具最小权限**：MCP Server 不暴露 promote/rollback 类写操作与黄金集内容——见 `evoflow/integrations/mcp_server.py`。

如果你发现这些红线存在绕过路径，这属于安全漏洞，请按上述流程报告。
