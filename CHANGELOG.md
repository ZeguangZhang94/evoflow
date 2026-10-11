# Changelog

本项目所有重要变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [0.1.0] - Unreleased

首次公开发布。

### Added

- **核心闭环**：三层进化栈 + 评估门，核心包零第三方依赖（纯标准库）。
  - L0 上下文进化：`Playbook` / `DeltaEntry` / `Reflector`（ACE 式增量 delta 策展）。
  - L1 经验库：`ReasoningBank`（成功+失败轨迹蒸馏，k=1 检索）/ `JsonlVectorStore`（char-3gram hashing 向量，零依赖）。
  - L2 数据飞轮：`SignalCollector` / `DataFlywheel`（`pos_neg_ratio=2.0`）/ `AdapterRegistry`（promote / rollback）。
  - 评估体系：`EvolutionGate`（`min_pass_rate=0.9`）/ `GoldenSet` / `RuleBasedJudge` / `LLMJudge`。
  - 迭代熔断：`IterationController`（`max_rounds=3`，超限抛 `IterationHalted`）。
  - CLI：`evoflow init / distill / gate / promote / rollback`。
- **共享 LLM 客户端 `evoflow.llm`**：零依赖（纯标准库 urllib）OpenAI 兼容客户端 `OpenAICompatClient`，可接百炼 / DeepSeek / vLLM 等任意兼容端点；`LLMClient` 协议；测试用 `ScriptedLLM`（无网络可测）。
- **L0 上下文进化**：
  - `llm_reflector.py`：`LLMReflector`（LLM 反思，解析失败回退为整条文本、LLM 异常返回空列表——绝不中断进化循环）+ `ACELoop`（Generator→Reflector→Curator 开箱闭环；红线：verifier 缺失即拒绝执行）。
  - `gepa.py`：`GEPAEvolver`（GEPA，[arXiv:2507.19457](https://arxiv.org/abs/2507.19457)：Pareto 前沿提示词进化；LLM 调用预算 `budget` 为一等硬约束，耗尽即停）。
- **L1 经验/记忆库**：
  - `episodic.py` `EpisodicMemory`（A-MEM 式卡片记忆，[arXiv:2502.12110](https://arxiv.org/abs/2502.12110)：入库自动与 top-k 相似笔记建链、互链笔记簇演化合并；LLM / 启发式双路）。
  - `profile.py` `ProfileMemory`（用户画像记忆，`forget()` 按 key 精确删除，满足合规删除场景）。
  - `qdrant_store.py` `QdrantStore`（生产级向量后端，extras `vec-qdrant`，惰性 import，核心包零依赖不变）。
- **知识层 `evoflow.knowledge`**：
  - `Document / Chunk / KnowledgeLayer` 协议。
  - `VectorRAG`：chunk + Contextual Prefix（Anthropic Contextual Retrieval 轻量实现）+ RRF(k=60) 混合检索 + `[来源:doc_id]` 引用溯源。
  - `LightGraphRAG`：双层检索（低层实体倒排 / 高层向量语义）、O(1) 增量入库；docstring 明确标注为**教学级简化实现**，生产建议替换为真 LightRAG / LazyGraphRAG。
  - `router.recommend()` 路由：语料 <200K token 且 <500 查询/天 → 长上下文；多跳 / 全局性查询 → 图；否则向量。
- **L2 训练后端**：
  - `qlora_dpo.py`：红线前置——`epochs > 3` / `rank > 32` 在配置期即拒绝；基于 TRL `DPOTrainer` + 4bit BitsAndBytes；`IterationController` 熔断接入训练回调；`mix_replay` 混入 10% 旧样本防灾难性遗忘。
    > ⚠️ 诚实声明：**训练主流程未经 GPU 实测**；配置校验、数据校验与熔断回调逻辑有单测覆盖，真实训练行为请先在自有硬件上小规模验证。
  - `echo_trap.py` `EchoTrapMonitor`（RAGEN Echo Trap 三信号，[arXiv:2504.20073](https://arxiv.org/abs/2504.20073)：奖励方差塌缩 / 熵降 / 梯度尖峰，任一触发即报警，`enforce()` 抛 `EchoTrapHalted` 停训）。
  - `serving.py` `VLLMLoRAServer`（评估门过线才能热切换 adapter；未晋升版本**零 HTTP 调用**即抛 `PromotionDenied`）。
- **可观测性 `evoflow.telemetry`**：`EvolutionEvent / EvolutionHistory`（9 类事件 jsonl 时间线，可回放整条进化轨迹）；`wrap_gate / wrap_registry` 组合式插桩（不改原类一行代码）；`OpenTelemetryExporter / LangfuseExporter`（extras `otel`，惰性 import，增量导出不重复）。
- **评测**：
  - `eval/calibration.py`：Cohen's Kappa 评委校准（`calibrate()` → `CalibrationReport`）、分层抽样送人工标注（`sample_for_human`）。
  - `eval/leakage.py`：黄金集防泄漏——精确匹配 + char-3gram Jaccard > 0.8 近重复双重检测；`assert_no_leakage` 供 `train()` 前 fail-fast。
- **MCP 集成 `evoflow.integrations.mcp_server`**：纯标准库 JSON-RPC stdio 模式（默认，零依赖）+ 官方 SDK 增强模式（extras `mcp`）；4 个工具：`evoflow_recall / evoflow_feedback / evoflow_reflect / evoflow_gate_status`。红线：不暴露黄金集、无 promote 类工具、`gate_status` 只读。
- **可选 extras**：`train`（QLoRA+DPO 训练栈）、`vec`、`vec-qdrant`、`llm`、`otel`、`mcp`、`dev`。
- 测试总量 214 个，全部通过。
