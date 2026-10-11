# EvoFlow

**简体中文** | [English](README_EN.md)

> 会自己变好的 LLM Agent 框架 —— 为中小企业而生。
> A self-evolving LLM agent framework that gets better from your own feedback signals — at coffee-money cost.

<!-- badges: start -->
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/evoflow/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-github%20actions-lightgrey)](.github/workflows/)
<!-- badges: end -->

**EvoFlow 不是一个"更好的模型"，而是一个"会自己变好的系统"**：知识放在参数之外（RAG），经验放在权重之外（playbook / 经验库），行为用 QLoRA 低成本固化，一切更新过评估门、可一键回滚。

---

## 为什么做 EvoFlow

大多数团队缺的不是模型，而是一套让 Agent **从真实使用中持续变好**的机制。EvoFlow 的设计建立在三组公开研究事实上：

| 关键事实 | 含义 | 来源 |
|---|---|---|
| **QLoRA 微调单次 $1–10** | 真要改权重时，成本已低到"一杯咖啡"量级（7B 级，租单卡 RTX 4090） | [Spheron, 2026-03](https://www.spheron.network/blog/how-to-fine-tune-llm-2026/) |
| **ACE 免训练 +10.6%** | 只维护一个增量进化的上下文 playbook，不改任何权重，AppWorld 任务成功率 +10.6%、适配延迟 -86.9% | [arXiv:2510.04618](https://arxiv.org/abs/2510.04618) |
| **评估门五件工程** | 自进化社区共识：采集轨迹 → 冻结 holdout 评估 → 单层更新提案 → 回归测试 → 版本化快照+回滚，缺一不可 | [AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/) |

一句话：**免训练进化（L0/L1）拿到第一波收益，低成本训练（L2）固化被反复验证的稳定行为，评估门保证"只变好、不变坏"。**

> ⚠️ 诚实声明：+10.6% 等为论文作者在公开基准上的自测数字，在你的业务场景上是否复现取决于反馈信号质量和基座强度。EvoFlow 内置的评估门正是为了让你用**自己的黄金集**验证收益，而不是相信别人的头条数字。

---

## 架构：三层进化栈 + 评估门

```
┌──────────────────────────────────────────────────────────────┐
│  评估门 EvolutionGate（冻结黄金集 + 回归测试，不过线不晋升）      │
│  评分器 / 黄金集对 agent 运行时不可见（防 reward hacking）        │
└──────────────▲────────────────────────────▲──────────────────┘
               │ 不过线即丢弃                │ 过线才晋升
   ┌───────────┴───────────┐   ┌────────────┴──────────────┐
   │ L0 上下文进化（实时）     │   │ L1 经验/记忆库（任务级）      │
   │ ACE 式 playbook：        │   │ ReasoningBank 式策略库：     │
   │ 增量 delta 策展          │   │ 成功+失败轨迹蒸馏，k=1 检索   │
   │ 成本：仅 LLM 调用费       │   │ 成本：≈$0                   │
   └───────────▲───────────┘   └────────────▲──────────────┘
               │ 稳定模式固化                │ 蒸馏入数据池
   ┌───────────┴────────────────────────────┴──────────────┐
   │ L2 权重层数据飞轮（周级迭代）                             │
   │ 反馈信号 → 偏好对 → QLoRA + DPO（单次 $1–10 量级）        │
   │ adapter 版本化，可 promote / rollback；基座永远冻结        │
   └────────────────────────────────────────────────────────┘
   ┌────────────────────────────────────────────────────────┐
   │ 知识层：VectorRAG / LightGraphRAG / 路由                 │
   │ 知识放在参数之外：按规模路由 长上下文 / 向量 / 图谱        │
   └────────────────────────────────────────────────────────┘
```

设计红线（每条都有公开研究证据，详见 [docs/architecture.md](docs/architecture.md)）：

1. **评估门是硬约束**：任何一层更新不过黄金集（默认 pass_rate ≥ 0.9）即丢弃。
2. **评分器隔离**：评分函数、验证器、黄金集对 agent 运行时永远不可见（METR 实测评分函数可见时作弊率相差 43 倍，[metr.org, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)）。
3. **迭代熔断**：自训练最多 2–3 轮（Self-Rewarding 第 4 轮饱和、ReST-EM 第 2 轮回退，见 [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) / [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)），超限抛 `IterationHalted`。
4. **黄金集不进训练数据**：飞轮导出与训练启动前自动做泄漏检测（`assert_no_leakage`）。

---

## 功能特性

EvoFlow 的完整能力清单如下，**214 个测试全绿**：

| 能力 | 模块 | 对应研究 / 依据 |
|---|---|---|
| 共享 LLM 客户端：零依赖 OpenAI 兼容客户端 + 协议 + 测试桩 | `evoflow.llm`（`OpenAICompatClient` / `LLMClient` / `ScriptedLLM`） | 工程基础件：百炼 / DeepSeek / vLLM 等任意兼容端点；纯标准库 urllib |
| ACE 开箱闭环：LLM 反思 + 三角色编排，verifier 缺失即拒跑 | `l0_context/llm_reflector.py`（`LLMReflector` / `ACELoop`） | ACE，[arXiv:2510.04618](https://arxiv.org/abs/2510.04618)（无可靠信号时自我更新退化） |
| GEPA Pareto 提示词进化，LLM 调用预算硬约束 | `l0_context/gepa.py`（`GEPAEvolver`） | GEPA，[arXiv:2507.19457](https://arxiv.org/abs/2507.19457)（论文自测：超 GRPO 约 10%、rollout 省约 35×） |
| A-MEM 式情节记忆：自动建链、演化合并 | `l1_memory/episodic.py`（`EpisodicMemory`） | A-MEM，[arXiv:2502.12110](https://arxiv.org/abs/2502.12110) |
| 用户画像记忆，可 `forget` 合规删除 | `l1_memory/profile.py`（`ProfileMemory`） | 知识放在参数之外 = 可删除、可审计 |
| Qdrant 生产向量后端 | `l1_memory/qdrant_store.py`（`QdrantStore`） | `VectorStore` 协议的替代实现（extras `vec-qdrant`） |
| 知识层：VectorRAG / LightGraphRAG / 路由 | `evoflow.knowledge` | Contextual Retrieval（[Anthropic, 2024-09](https://www.anthropic.com/news/contextual-retrieval)）、LightRAG（[arXiv:2410.05779](https://arxiv.org/abs/2410.05779)）、混合检索共识 |
| L2 训练后端生产化：配置期红线 + 熔断回调 | `l2_weight/trainers/qlora_dpo.py` | QLoRA（[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)）；⚠️ **训练主流程未经 GPU 实测** |
| Echo Trap 训练塌缩监控，报警即停 | `l2_weight/trainers/echo_trap.py`（`EchoTrapMonitor`） | RAGEN，[arXiv:2504.20073](https://arxiv.org/abs/2504.20073) |
| vLLM LoRA 热切换：未过评估门零 HTTP 调用即拒 | `l2_weight/serving.py`（`VLLMLoRAServer`） | 评估门过线即上线的工程闭环 |
| 进化可观测性：9 类事件 jsonl 时间线 + 组合插桩 | `evoflow.telemetry` | 工程件；OTel / Langfuse 导出为 extras `otel` |
| 评委校准（Cohen's Kappa）+ 黄金集防泄漏 | `eval/calibration.py` / `eval/leakage.py` | LLM 评委与人工一致率有限，上线前需校准（Zheng et al., [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)） |
| MCP Server：4 个工具，stdio 模式零依赖 | `evoflow.integrations.mcp_server` | MCP 协议（2025-06-18）；不暴露黄金集、无 promote 工具 |

> ⚠️ 诚实声明：GEPA "超 GRPO 10%、rollout 省 35×" 为论文作者自测数字；`LightGraphRAG` 是教学级简化实现（docstring 已标注）；**L2 训练主流程未经 GPU 实测**，配置/数据校验与熔断回调有单测覆盖，请先小规模验证再上量。

---

## 安装

核心包**零第三方依赖**（纯标准库），训练/向量/LLM/可观测能力为可选 extras：

```bash
# 最小安装：L0 playbook + L1 内置 JsonlVectorStore（char-ngram hashing 向量，无需嵌入模型）
pip install evoflow

# 全都要
pip install "evoflow[train,vec,vec-qdrant,llm,otel,mcp]"
```

| Extra | 依赖 | 用途 |
|---|---|---|
| （无） | 零第三方依赖 | L0/L1/评估门/知识层/MCP stdio 模式全闭环，开箱即用 |
| `train` | torch / trl / peft / datasets | L2 训练后端（QLoRA + DPO） |
| `vec` | numpy | 向量运算支持 |
| `vec-qdrant` | qdrant-client | `QdrantStore` 生产向量后端 |
| `llm` | openai | OpenAI 官方 SDK；注：核心 `OpenAICompatClient` 纯标准库实现，不装此 extra 也能连兼容端点 |
| `otel` | opentelemetry-sdk / langfuse | 进化事件流水导出到可观测平台 |
| `mcp` | mcp | MCP Server 官方 SDK（FastMCP）增强模式 |
| `dev` | pytest | 跑测试 |

要求 Python ≥ 3.10。

---

## 5 分钟快速开始

下面的例子只用到核心包（零依赖），跑通"任务 → 反馈 → 反思 → playbook 进化 → 评估门"的完整闭环。

### 1. L0：上下文进化（ACE 式 playbook）

```python
from evoflow.core.trajectory import Trajectory, Outcome
from evoflow.l0_context.playbook import Playbook
from evoflow.l0_context.reflector import Reflector

# 1) 定义一个反思函数：输入轨迹，输出增量条目（生产环境换成你的 LLM 调用）
def my_reflect(traj: Trajectory):
    if not traj.outcome.success:
        return [{"title": "报价前必须确认税率", "content": "含税/不含税需先问清",
                 "tags": ["pricing"], "helpful": 1}]
    return []

reflector = Reflector(reflect_fn=my_reflect)
playbook = Playbook()

# 2) 一次失败任务 → 反思 → 策展进 playbook
traj = Trajectory(task="给客户报价", steps=["直接报了不含税价"],
                  outcome=Outcome(success=False, verifier="user", detail="客户投诉"))
playbook.curate(reflector.reflect(traj))

# 3) 下次任务注入 prompt
system_prompt = playbook.render(max_tokens=800)

# 4) 持久化
playbook.save("playbook.json")
```

### 2. L1：经验库（成功+失败轨迹蒸馏，k=1 检索）

```python
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore

# JsonlVectorStore：内置 char-ngram hashing 向量，零依赖，开箱即用
bank = ReasoningBank(store=JsonlVectorStore("memory.jsonl"), embed_fn=None)

bank.distill(traj)  # 成功和失败的轨迹都蒸馏成 StrategyItem

# 新任务开始前检索 1 条最相关策略注入 prompt（k=1 是 ReasoningBank 论文的最优设置）
strategies = bank.retrieve("处理客户报价异议", k=1)
```

### 3. 评估门：不过线不晋升

```python
from evoflow.core.gate import EvolutionGate
from evoflow.eval.golden_set import GoldenSet
from evoflow.eval.judge import RuleBasedJudge

golden = GoldenSet.from_jsonl("golden.jsonl")   # 字段：input / expected
gate = EvolutionGate(golden_set=golden, judge=RuleBasedJudge(), min_pass_rate=0.9)

# candidate：任何 (str) -> str 的候选系统，比如"基座 prompt + 新 playbook"
result = gate.evaluate(candidate=my_agent_fn)
print(result.passed, result.pass_rate)   # False 则丢弃这次更新
```

### 4. L2：数据飞轮 → QLoRA + DPO（可选，需 `evoflow[train]`）

```python
from evoflow.core.signals import SignalCollector
from evoflow.l2_weight.flywheel import DataFlywheel
from evoflow.l2_weight.registry import AdapterRegistry

flywheel = DataFlywheel(collector=SignalCollector())
pairs = flywheel.build_pairs(signals, trajectories, pos_neg_ratio=2.0)
flywheel.export_dpo_jsonl("dpo_data.jsonl")

# 训练（pip install evoflow[train]），配置内置熔断上限
from evoflow.l2_weight.trainers.qlora_dpo import QLoRADPOConfig, train
config = QLoRADPOConfig(model_name="Qwen/Qwen3-8B", rank=16, epochs=2,
                        lr=1e-5, replay_ratio=0.1, max_rounds=3)
train(config, "dpo_data.jsonl")

# adapter 版本化：过评估门才晋升，不过线一键回滚
registry = AdapterRegistry(root="./adapters")
registry.register("r1", path="./adapters/r1")
if result.passed:
    registry.promote("r1")
else:
    registry.rollback()
```

---

## CLI

```bash
evoflow init        # 初始化项目骨架（golden.jsonl、playbook.json、目录结构）
evoflow distill     # 从轨迹日志蒸馏策略 / 构建 DPO 偏好对
evoflow gate        # 用冻结黄金集评估候选系统，输出 GateResult
evoflow promote     # 晋升指定 adapter 版本为当前生产版本
evoflow rollback    # 回滚到上一个通过评估门的版本
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录（Keep a Changelog 格式） |
| [docs/architecture.md](docs/architecture.md) | 三层进化栈详细设计、数据流图、设计红线与研究证据 |
| [docs/research-to-code.md](docs/research-to-code.md) | 研究结论 → 代码模块对照表（每条设计决策落在哪个类/硬约束） |
| [docs/integration.md](docs/integration.md) | LangGraph / CrewAI / Dify 集成，Qdrant、OpenAI 兼容 API 接入 |
| [docs/open-source-roadmap.md](docs/open-source-roadmap.md) | 开源发布、CI、版本策略、社区运营路线 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 |

## 路线图

**当前状态**：功能完整 —— 三层进化栈、评估门、知识层、训练后端、可观测性与 MCP 集成全部就位，214 个测试全绿，核心包零第三方依赖。诚实说明：L2 训练主流程尚未经 GPU 实测，配置/数据校验与熔断回调有单测覆盖，请先在自有硬件上小规模验证。

**通往 1.0 的方向**：

- **生态适配器**：LangGraph / CrewAI / Dify 适配器，更多 RAG / GraphRAG 集成示例
- **GPU 实测**：L2 训练主流程端到端 GPU 验证与多后端评测
- **文档完善**：完整文档与 API 稳定化，而后发布 1.0

详见 [docs/open-source-roadmap.md](docs/open-source-roadmap.md)。

## 引用与致谢

EvoFlow 的设计直接建立在以下公开研究之上：

- ACE: Agentic Context Engineering — [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- ReasoningBank — [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
- GEPA: Reflective Prompt Evolution — [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- A-MEM: Agentic Memory — [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- QLoRA — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- RAGEN / StarPO（Echo Trap）— [arXiv:2504.20073](https://arxiv.org/abs/2504.20073)
- Self-Rewarding Language Models — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
- LightRAG — [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
- METR reward hacking 实测 — [metr.org, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
- 自进化 Agent 综述 — [arXiv:2507.21046](https://arxiv.org/abs/2507.21046)

如果你在研究或产品中使用了 EvoFlow，欢迎引用本仓库（见 [CITATION.cff](CITATION.cff)）。

## License

[Apache-2.0](LICENSE)

---