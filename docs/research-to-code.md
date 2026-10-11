# 研究证据 → 代码对照表

> EvoFlow 与普通 agent 框架的核心区别：**框架里的每个类、每个默认参数、每条硬约束，都对应一条可溯源的公开研究证据**——不是"我们觉得这样好"，而是"论文/工业实测证明必须这样"。
>
> 下表"来源"列只引用公开一手来源：arXiv 论文、官方博客（Anthropic / METR / Cursor 等）、公开综述。标注"工程决策"的条目是框架自身的工程设计判断，不是外部研究结论。

---

## 一、核心抽象层（`evoflow.core`）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 自进化闭环的第一步是采集带可验证结果的轨迹；没有可靠反馈信号时自我更新会退化 | ACE 论文警示，[arXiv:2510.04618](https://arxiv.org/abs/2510.04618) | `Trajectory(task, steps, outcome)`；`Outcome(success, verifier, detail)` 强制记录"谁验证的" |
| 用户接受/拒绝是最廉价的奖励信号（Cursor Tab：接受 +0.75 / 拒绝 -0.25，4 亿请求/天，1.5–2 小时滚一个 checkpoint） | [Cursor 官方博客, 2025-09](https://cursor.com/blog/tab-rl) | `SignalType.ACCEPT / REJECT`；`FeedbackSignal(type, task_id, value, meta)` 的 `value` 即奖励标量 |
| 任务成败是免训练进化的前提信号 | ACE 论文推论，[arXiv:2510.04618](https://arxiv.org/abs/2510.04618) | `SignalType.SUCCESS / FAIL` |
| 自进化 = 写入路径 + 评估门五件工程：采集轨迹 → 冻结 holdout 评估 → 单层更新提案 → 回归测试 → 版本化快照+回滚 | [AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/) | `EvolutionGate(golden_set, judge, min_pass_rate=0.9).evaluate(candidate) -> GateResult(passed, pass_rate, failures, decision)` + `AdapterRegistry` 的回滚能力 |
| 自我迭代 2–3 轮后饱和：Self-Rewarding 第 4 轮饱和、ReST-EM 第 2 轮回退 | [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)；[arXiv:2312.06585](https://arxiv.org/abs/2312.06585) | `IterationController(max_rounds=3).record_round(metric)`，超限抛 `IterationHalted`——**熔断是异常不是警告，无法静默忽略** |

## 二、L0 上下文进化（`evoflow.l0_context`）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 只维护进化的上下文 playbook，免训练即可 AppWorld +10.6%、适配延迟 -86.9%（论文自测） | [arXiv:2510.04618](https://arxiv.org/abs/2510.04618) | `Playbook` 整个模块的存在理由 |
| 增量 delta 更新而非整段重写（防 context collapse） | 同上 | `Playbook.curate(deltas)` 接受增量条目列表；`DeltaEntry` 为最小更新单元 |
| 条目带 helpful/harmful 信号用于剪枝 | 同上 | `DeltaEntry.helpful` 计数字段 |
| Generator→Reflector→Curator 三角色中，Reflector 负责从轨迹提炼经验 | 同上 | `Reflector(reflect_fn).reflect(trajectory) -> list[DeltaEntry]`；`reflect_fn` 由用户注入——框架不绑定模型，LLM 是可选能力 |
| 上下文有 token 预算，超长 playbook 必须裁剪 | 工程常识 + ACE 的 items-as-tokens 管理 | `Playbook.render(max_tokens)` |
| **（v0.2）** ACE 三角色循环开箱即用；没有可靠反馈信号时自我更新反而退化——无 verifier 必须拒绝执行 | [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)（论文明确警示） | `ACELoop.run()` 缺 verifier 直接 `raise ValueError`（红线）；`LLMReflector` 解析失败回退为整条文本、LLM 异常返回空列表——绝不中断进化循环 |
| **（v0.2）** GEPA：用自然语言反思替代标量奖励做突变，Pareto 前沿维护候选种群；论文自测同等预算超 GRPO 约 10%、rollout 省约 35×——LLM 调用预算必须是一等硬约束 | [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) | `GEPAEvolver(budget=...)` 调用预算耗尽即停；只接受 `eval_fn` callable，不访问 judge 内部实现（评分器隔离）；`pareto_front` 纯函数可独立测试 |

## 三、L1 经验/记忆库（`evoflow.l1_memory`）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 从成功**和失败**轨迹蒸馏结构化推理策略（失败轨迹同等重要） | [arXiv:2509.25140](https://arxiv.org/abs/2509.25140) | `ReasoningBank.distill(trajectory)` 接受任意 `Outcome`，成败都入库 |
| **k=1 最优**：只取 1 条最相关记忆，相关性 > 数量 | 同上 | `ReasoningBank.retrieve(query, k=1)`——默认值即论文结论 |
| 策略条目为"标题 + 描述 + 正文"结构，标题做检索锚点 | 同上 | `StrategyItem(title, description, content)` |
| 收益口径（论文自测）：WebArena +8.3pp、SWE-bench-V +3.4pp、交互步数 -16% | 同上 | 本框架的 L1 预期收益参考区间（需自有黄金集复测） |
| 知识放参数之外 = 零训练成本、即时生效、可溯源、可删除；参数编辑方法（ROME/MEMIT）几百次更新即崩溃，RAG 全面胜出 | WikiBigEdit，[arXiv:2503.05683](https://arxiv.org/abs/2503.05683) | 经验库整体设计为外部 JSONL/向量存储，不进权重；`VectorStore` 是 Protocol 而非具体实现，可换后端 |
| 中小规模应零依赖起步；记忆框架头条数字多为厂商自测，必须用自有语料复测 | 工程决策（零依赖承诺） | 默认 `JsonlVectorStore`（char-3gram hashing 向量）零依赖；文档明确标注其语义检索弱于真嵌入模型 |
| **（v0.2）** A-MEM：Zettelkasten 卡片式记忆——入库自动建链、互链笔记簇演化合并（具体情节 → 抽象图式）；论文/厂商增益数字基于其自有基准，不可全信 | [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) | `EpisodicMemory` / `MemoryNote`：关键词/标签/上下文结构化卡片，自动与 top-k 相似笔记互链；LLM / 启发式双路；docstring 要求用自有语料复测建链密度与演化阈值 |
| **（v0.2）** 用户事实（偏好、约束、背景）不进权重，按 key 精确删除满足隐私合规 | WikiBigEdit（"知识放参数之外 = 可删除"），[arXiv:2503.05683](https://arxiv.org/abs/2503.05683) | `ProfileMemory.forget(key)`；`render()` 输出可直接注入 system prompt |
| **（v0.2）** 记忆库上规模后接生产向量库，但不破坏零依赖承诺 | 工程决策（零依赖承诺的推论） | `QdrantStore` 实现 `VectorStore` 协议（extras `vec-qdrant`）；`qdrant_client` 构造时才惰性 import，核心包零依赖不变 |

## 四、L2 权重层（`evoflow.l2_weight`）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| QLoRA 微调单次 $1–10（7B 级，租单卡 4090）；托管备选百炼 ¥180/次 | [Spheron, 2026-03](https://www.spheron.network/blog/how-to-fine-tune-llm-2026/) | `QLoRADPOConfig(model_name, rank=16, epochs=2, lr=1e-5, ...)` 的参数区间按此成本量级设计；`train()` 为可选 extra |
| 偏好对齐用 DPO 而非 SFT：RFT/DPO 类天然缓解灾难性遗忘，SFT 导致旧技能严重退化 | [arXiv:2507.05386](https://arxiv.org/abs/2507.05386) | 训练器命名为 `qlora_dpo`；`export_dpo_jsonl()` 只导出偏好对格式 |
| 基座永远冻结，进化全在 adapter；adapter 全量版本化，过评估门才热切换，不过线即丢弃 | [AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/)（版本化快照+回滚） | `AdapterRegistry(root).register()/.promote(version)/.rollback()`；promote 动作与 `GateResult.passed` 绑定是推荐用法 |
| 自训练迭代上限 2–3 轮 | [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)；[arXiv:2312.06585](https://arxiv.org/abs/2312.06585) | `QLoRADPOConfig.max_rounds=3`，与 `IterationController` 双重保险 |
| Model Collapse 风险：递归用生成数据训练导致不可逆退化；缓解靠累积真实数据 + 验证器过滤 | Nature 2024，[DOI 10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y) | `build_pairs` 只从**真实反馈信号 + 真实轨迹**构建偏好对，不接收纯模型自生成数据；`replay_ratio` 保证旧真实数据不被替换 |
| **（v0.2）** 训练红线配置期前置：不要等到训练跑起来才发现配置违规 | 工程决策（fail-fast） | `QLoRADPOConfig` 在配置期即拒绝 `epochs>3` / `rank>32`；基于 TRL `DPOTrainer` + 4bit BitsAndBytes；`IterationController` 熔断接入训练回调；`mix_replay` 混 10% 旧样本 |
| **（v0.2）** Echo Trap：多轮 agent RL 中奖励方差塌缩 → 熵降 → 梯度尖峰 → 策略同质化，报警即停，继续训练只会浪费算力放大漂移 | RAGEN，[arXiv:2504.20073](https://arxiv.org/abs/2504.20073) | `EchoTrapMonitor` 三信号检测，`enforce()` 报警即抛 `EchoTrapHalted`，供训练循环每步调用 |
| **（v0.2）** 评估门过线即上线：adapter 热切换无需重启推理服务；未过门绝不能触网 | [AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/)（过评估门才上线） | `VLLMLoRAServer.promote_and_swap()` 复用 `AdapterRegistry.promote` 校验，未晋升版本**零 HTTP 调用**即抛 `PromotionDenied` |
| **（v0.2）** 诚实声明：训练主流程未经 GPU 实测 | — | 文档与 docstring 明确标注；配置校验、数据校验、熔断回调逻辑有单测覆盖，真实训练请先小规模验证 |

## 五、评估体系（`evoflow.eval`）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 冻结 holdout 金标样本是一切优化的闸门 | [AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/)（冻结 holdout 评估） | `GoldenSet.from_jsonl(path)`，字段 `input/expected`；`evoflow init` 生成模板 |
| 评估门默认高门槛（不过线不晋升） | 同上（五件工程） | `EvolutionGate(min_pass_rate=0.9)` |
| 规则判分确定性、零成本，优先于 LLM 判分 | 工程共识 | `RuleBasedJudge` 内置；`Judge` 为 Protocol |
| 评分函数对 agent 可见会诱发 43 倍差异的 reward hacking（30.4% vs 0.7%）；生产 RL 中学会 hacking 会泛化为广谱错位 | [METR, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)；[arXiv:2511.18397](https://arxiv.org/abs/2511.18397) | **架构级隔离**：`evoflow.eval` 与 agent 运行路径（l0/l1）无依赖关系；`Outcome.verifier` 只存标签不存逻辑；黄金集禁止进训练数据（`GateResult.failures` 仅供人工诊断） |
| **（v0.2）** LLM 评委上线前必须人工校准；人工标注成本高，需抽样控制 | 工程共识 | `calibrate(judge_scores, human_scores) -> CalibrationReport`（Cohen's Kappa，纯标准库）；`sample_for_human` 分层随机抽样送人工 |
| **（v0.2）** 黄金集绝不能进训练数据（红线 4）；近重复改写也算泄漏 | 评分器隔离原则推论；[METR, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/) | `check_leakage` / `assert_no_leakage`：规范化精确匹配 + char-3gram Jaccard > 0.8 近重复双重检测，覆盖黄金集 input/expected 与训练对 task/chosen/rejected 全部字段；`train()` 前 fail-fast |

## 六、共享 LLM 客户端（`evoflow.llm`，v0.2 新增）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 框架不绑定模型，但各进化模块（Reflector / LLMJudge / GEPA / 知识层）不应各写一套 HTTP 逻辑 | 工程决策（零依赖承诺的推论） | `OpenAICompatClient`：纯标准库 urllib 实现 OpenAI 兼容 Chat Completions，可接百炼 / DeepSeek / vLLM 等任意兼容端点；`LLMClient` Protocol 统一注入点；`ScriptedLLM` 供无网络测试 |

## 七、知识层（`evoflow.knowledge`，v0.2 新增）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 混合检索（关键词 + 向量）融合优于单一检索；RRF(k=60) 是免调参的经典融合默认 | RRF 原论文（Cormack et al., SIGIR 2009, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*） | `VectorRAG(hybrid=True)`（默认）：零依赖词频匹配（BM25 替身）+ 向量相似度，RRF(k=60) 融合 |
| Contextual Retrieval：为每个 chunk 补一段"它在原文中的位置/背景"前缀，显著降低检索语境丢失 | [Anthropic 官方博客, 2024-09](https://www.anthropic.com/news/contextual-retrieval) | `chunker.ContextualPrefixer`：有 LLM client 时生成 50–100 token 前缀，无 client 退化为启发式（metadata 标题 + 文档首句） |
| LightRAG 双层检索：低层精确事实（实体）/ 高层宽泛主题（语义） | [arXiv:2410.05779](https://arxiv.org/abs/2410.05779) | `LightGraphRAG`：实体 → chunk 倒排 + 向量语义双层检索，O(1) 增量入库；**docstring 明确标注为教学级简化实现**，生产建议替换为真 LightRAG / LazyGraphRAG |
| 检索决策线：语料 <200K token 且 <500 查询/天 → 直接长上下文；多跳/全局查询 → 图；否则向量 | 工程经验法则（启发式默认值，可按业务调整） | `router.recommend(corpus_tokens, queries_per_day, multi_hop)` 返回 `long_context / graph / vector` |
| 检索结果必须可溯源到具体文档 | 工程共识 | `VectorRAG` 输出带 `[来源:doc_id]` 引用 |

## 八、可观测性（`evoflow.telemetry`，v0.2 新增）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 自进化系统的每次更新都必须可回放、可审计（五件工程的"版本化快照"在事件层的体现） | [AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/) | `EvolutionEvent / EvolutionHistory`：信号采集、反思、蒸馏、偏好对、训练轮、评估门裁决、晋升、回滚、熔断 9 类事件追加 jsonl 时间线 |
| 组合优于继承：插桩不应改动被观测类的一行代码 | 工程决策 | `wrap_gate / wrap_registry` 返回包装对象拦截 `evaluate / promote / rollback` 自动记录事件，其余属性经 `__getattr__` 透传 |
| 可观测性平台是可选能力，不能污染零依赖核心 | 零依赖承诺 | `OpenTelemetryExporter / LangfuseExporter` 惰性 import（extras `otel`）；记录已导出最大 ts，重复调用只做增量导出 |

## 九、MCP 集成（`evoflow.integrations.mcp_server`，v0.2 新增）

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 进化能力应能被任意 MCP 宿主（Claude Code / Cursor 等）直接调用 | MCP 协议（2025-06-18） | 4 个工具：`evoflow_recall` / `evoflow_feedback` / `evoflow_reflect` / `evoflow_gate_status`，均为现有 API 的薄封装 |
| 工具化不能破红线：黄金集不暴露、晋升权不外放 | 评分器隔离原则（红线 2）推论 | 无 promote 类工具；`gate_status` 只读；黄金集对 MCP 工具不可见 |
| 零依赖承诺覆盖集成层 | 零依赖承诺 | 默认纯标准库 JSON-RPC stdio 模式；官方 SDK（FastMCP）增强模式惰性 import（extras `mcp`），工具集两模式完全一致 |

## 十、CLI（`evoflow` 命令行）

| 研究结论 | 落在代码上 |
|---|---|
| 阶段 0 动作：建金标评测集 → 部署经验库 → 设计反馈信号 | `evoflow init` 生成 golden.jsonl / playbook.json 骨架 |
| 飞轮动作：日志 → 偏好对 → 蒸馏策略 | `evoflow distill` |
| 五件工程：冻结 holdout 评估 → 回归测试 | `evoflow gate` |
| 五件工程：版本化快照 + 回滚 | `evoflow promote` / `evoflow rollback` |

## 十一、框架级设计决策

| 研究结论 | 来源 | 落在代码上 |
|---|---|---|
| 成本-可逆性光谱：90% 的自我完善放在光谱外侧，只有被反复验证的稳定行为才向权重固化 | [xinmingtu.github.io, 2026-06](https://xinmingtu.github.io/blog/2026/self-evolving-agents/) | 三层包结构 `l0_context / l1_memory / l2_weight` 本身就是光谱的代码化；L2 依赖最重且为可选 |
| 越靠外越频繁：L0 实时、L1 天级、L2 周级 | 成本-可逆性光谱的工程推论（同上） | 各层 API 的调用成本与持久化粒度设计（playbook 每任务 render；bank 任务级 distill；adapter 周级 train） |

---

## 如何读这张表

- **选型尽调**：如果你在评估 EvoFlow，每一行都是一次"框架设计是否有依据"的审计入口——从代码反查到论文。
- **贡献代码**：新增功能前请先回答"它对应哪条研究结论？"——答不上来的功能大概率不该进核心包。详见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
- **数字免责**：表中所有性能/成本数字的审计状态（论文自测 / 第三方复现 / 工程经验法则）均已在来源列标注，引用时请勿去掉标注。
