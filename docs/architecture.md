# EvoFlow 架构设计文档

> 本文档解释 EvoFlow 的"三层进化栈 + 评估门"架构：每层职责、数据流，以及三条设计红线背后的研究证据。
> 所有关键结论可溯源到公开论文与工业实测（arXiv / 官方博客 / 公开综述，见文末"参考"）。

---

## 1. 设计哲学：成本-可逆性光谱

工程界对"自进化"已形成一条共识光谱：

```
外部文件（记忆/技能库） → Agent Harness（提示词/工具/工作流） → 模型权重
越靠外越便宜、即时、可逆；越靠里越贵、慢、持久。
```

EvoFlow 的战略含义（[自进化 agent 工程综述, 2026-06](https://xinmingtu.github.io/blog/2026/self-evolving-agents/)）：**把 90% 的"自我完善"放在光谱外侧，只有被反复验证的稳定行为模式才允许向权重层固化。** 三层进化栈就是这条光谱的代码化：

| 层 | 模块 | 改什么 | 频率 | 单次成本 | 可逆性 |
|---|---|---|---|---|---|
| L0 上下文进化 | `evoflow.l0_context` | playbook（增量 delta） | 每次任务 | 仅 LLM 调用费 | 完全可逆（JSON 文件） |
| L1 经验/记忆库 | `evoflow.l1_memory` | 蒸馏的策略条目 | 任务级沉淀 | ≈$0（默认 hashing 向量） | 完全可逆 |
| L2 权重层 | `evoflow.l2_weight` | QLoRA adapter | 周级迭代 | $1–10 / <¥10 | adapter 版本化，可回滚 |

基座模型**永远冻结**：LoRA 冻结基座既防灾难性遗忘，也让"换基座"成本极低——知识在 RAG、经验在记忆库、行为在 adapter，基座是可替换商品。

---

## 2. L0：上下文进化（`evoflow.l0_context`）

### 职责

维护一个随任务增量进化的上下文 playbook，注入每次调用的 system prompt。**不改任何权重**。

### 组件

| 类 | 职责 |
|---|---|
| `Playbook` | 结构化上下文容器。`.curate(deltas)` 做增量条目策展（合并/去重/更新 helpful 计数），`.render(max_tokens)` 按 token 预算渲染进 prompt，`.save()` / `.load()` JSON 持久化 |
| `DeltaEntry` | 单条增量条目：title / content / tags / helpful 计数 |
| `Reflector` | 反思器。`.reflect(trajectory) -> list[DeltaEntry]`，反思逻辑由用户注入的 `reflect_fn` 决定（可以是规则，也可以是 LLM 调用——框架不绑定任何模型） |

### 对应研究：ACE（arXiv:2510.04618）

ACE（斯坦福等，2025-10，ICLR 2026）用 Generator → Reflector → Curator 三角色维护进化中的 playbook，关键设计被 EvoFlow 逐条吸收：

- **增量 delta 更新而非整段重写**——ACE 论文的核心发现：整体重写会导致"上下文塌缩"（context collapse），增量条目才可持续进化。→ 对应 `Playbook.curate(deltas)` + `DeltaEntry`。
- **条目带 helpful/harmful 计数**，用于后续剪枝。→ `DeltaEntry.helpful`。
- **免训练收益**：AppWorld +10.6%、适配延迟 -86.9%、每任务约 3 次 LLM 调用（论文自测数字）。

### 警示（论文明确证据）

ACE 论文指出：**没有可靠反馈信号（GT 标签或执行结果）时，自我更新反而退化**——上下文被伪信号污染。这就是 `FeedbackSignal` / `Outcome.verifier` 存在的意义：L0 的反思输入必须挂在可验证的任务结果上，而不是模型的自我感觉。另外 Dynamic Cheatsheet（arXiv:2504.07952）证明弱基座上记忆进化收益很小甚至为负——**L0 要求 7B 级以上的现代基座**。

---

## 3. L1：经验/记忆库（`evoflow.l1_memory`）

### 职责

把任务轨迹（成功**和失败**）蒸馏成结构化策略条目，向量检索后注入新任务的 prompt。与 L0 的区别：L0 是"当前最佳上下文"的单一 playbook；L1 是"按查询检索"的策略仓库。

### 组件

| 类 | 职责 |
|---|---|
| `ReasoningBank` | `.distill(trajectory)` 把轨迹蒸馏为 `StrategyItem(title, description, content)`；`.retrieve(query, k=1)` 检索注入 |
| `StrategyItem` | 策略条目：标题（检索锚点）、描述、正文 |
| `VectorStore`（Protocol） | 向量存储协议，可接 Qdrant 等后端 |
| `JsonlVectorStore` | 默认实现：JSONL 文件 + char-3gram hashing 向量，**零依赖**，适合中小规模（数千条）开箱即用 |

### 对应研究：ReasoningBank（arXiv:2509.25140）

Google 等（2025-09，ICLR 2026）的核心发现逐条落在代码上：

- **成功和失败轨迹都要蒸馏**——失败轨迹蒸馏出的"避坑策略"与成功经验同等重要。→ `distill()` 不区分成败，由 `Outcome` 决定蒸馏视角。
- **k=1 最优**：只取 1 条最相关记忆注入，相关性 > 数量。→ `.retrieve(query, k=1)` 的默认参数就是论文结论，不是随意选的。
- **收益口径**（论文自测）：WebArena +8.3pp、SWE-bench-V +3.4pp、交互步数 -16%。

### 成本

默认 `JsonlVectorStore` 不调用任何嵌入模型，存储与检索成本为零。接入外部嵌入/向量库（见 [integration.md](integration.md)）后，单条记忆操作成本 <$0.001（参考 A-MEM 的 <$0.0003/次口径，arXiv:2502.12110）。

---

## 4. L2：权重层数据飞轮（`evoflow.l2_weight`）

### 职责

把 L0/L1 运行中积累的反馈信号转成偏好对，用 QLoRA + DPO 低成本固化到 adapter；adapter 全量版本化，过评估门才晋升。

### 组件

| 类 | 职责 |
|---|---|
| `SignalCollector` / `FeedbackSignal` | 反馈采集。`SignalType ∈ ACCEPT / REJECT / SUCCESS / FAIL / CORRECTION`——接受/拒绝模式来自 Cursor Tab 的工业实践（见下） |
| `DataFlywheel` | `.build_pairs(signals, trajectories, pos_neg_ratio=2.0)` 构建偏好对；`.export_dpo_jsonl(path)` 导出训练数据 |
| `PreferencePair` | `(chosen, rejected, task)` |
| `QLoRADPOConfig` / `train()` | 训练入口（需 `evoflow[train]`）。默认 rank=16、epochs=2、lr=1e-5、replay_ratio=0.1、max_rounds=3 |
| `AdapterRegistry` | adapter 版本化：`.register()` / `.promote(version)` / `.rollback()` |

### 默认值背后的证据（每个参数都能溯源）

- **`pos_neg_ratio=2.0`**：反馈降噪工程实践的正负样本比 2:1（工程经验值，未审计，仅作量级参考）。
- **`replay_ratio=0.1`**：混入 10% 旧样本防灾难性遗忘。
- **`rank=16`（≤32）、`epochs=2`（≤3）**：LoRA 防遗忘工程共识；LoRA 只训 0.1%–1% 参数即可恢复全量微调 90–95% 质量，且"learns less and forgets less"。
- **单次成本 $1–10**：7B 级 QLoRA，租单卡 RTX 4090（$0.14–0.89/小时）（Spheron: 2026-03；Vast.ai 价格快照）；托管备选阿里云百炼 ¥180/次。
- **50 条高质量 DPO 样本即可见收益**（+30%，CSDN 工程文，未审计）——所以飞轮不需要大数据量，需要的是高质量反馈信号。

### 正样本优先级（build_pairs 的排序依据）

用户纠正（CORRECTION）> 知识库检索 > 强模型蒸馏 > 人工标注（工程实践）。`SignalType.CORRECTION` 单独成枚举就是为了承载这条最高价值信号。

---

## 5. 数据流全景

```
                    ┌─────────────────────────────────────────┐
                    │  业务流程（必须能产生可验证结果）           │
                    │  用户接受/拒绝 · 任务成败 · 单测通过        │
                    └──────────────────┬──────────────────────┘
                                       │ FeedbackSignal / Trajectory
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  SignalCollector（反馈信号落库）                                    │
└───────┬───────────────────────────────┬──────────────────────────┘
        │                               │
        ▼                               ▼
┌───────────────┐             ┌──────────────────┐
│ L0 Reflector   │             │ L1 ReasoningBank  │
│ → DeltaEntry   │             │ → StrategyItem    │
│ → Playbook     │             │ → k=1 检索注入     │
└───────┬───────┘             └────────┬─────────┘
        │  playbook 进化                │ 策略库沉淀
        └───────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │ DataFlywheel      │
              │ build_pairs()     │  ← pos_neg_ratio=2.0
              │ export_dpo_jsonl()│
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ train()           │  ← QLoRA + DPO（evoflow[train]）
              │ 新 adapter 候选    │
              └────────┬─────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  EvolutionGate.evaluate(candidate)│ ← 冻结黄金集，min_pass_rate=0.9
        └───────┬──────────────────┬───────┘
                │ passed=False     │ passed=True
                ▼                  ▼
        ┌──────────────┐   ┌────────────────┐
        │ 丢弃候选       │   │ registry.promote│
        │ （或 rollback）│   │ 热切换生产版本   │
        └──────────────┘   └────────────────┘

  IterationController 贯穿 L2 全程：record_round() 计数，
  超过 max_rounds（默认 3）抛 IterationHalted，强制停止。
```

---

## 6. 三条设计红线（硬约束，不是建议）

### 红线 1：评估门是硬约束 —— `EvolutionGate`

**规则**：冻结黄金集（`GoldenSet.from_jsonl`，input/expected 字段）+ judge，默认 `min_pass_rate=0.9`。任何一层（L0 的 playbook 版本、L1 的策略库、L2 的 adapter）的候选更新不过线即丢弃，过线才晋升。`GateResult.failures` 记录失败用例用于诊断，但**黄金集本身永远不进训练数据**。

**证据**：

- 自进化社区共识"五件工程"：采集轨迹 → 冻结 holdout 评估 → 单层更新提案 → 回归测试 → 版本化快照+回滚（[AgentsPulse 综述, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/)）。
- 没有可靠反馈信号的自我更新会退化（ACE 论文明确警示，[arXiv:2510.04618](https://arxiv.org/abs/2510.04618)）。
- 工程口径：50–100 条金标样本是"一切优化的闸门"；LLM-as-judge 上线前需人工校准——所以 EvoFlow 同时提供 `RuleBasedJudge`（确定性、免费）和 `LLMJudge`（语义判断、需校准）。

### 红线 2：评分器对 agent 不可见 —— 架构级隔离

**规则**：`judge`、黄金集、验证器只存在于评估侧（`evoflow.eval` / `EvolutionGate`），任何 agent 运行路径（L0 render、L1 retrieve、推理调用）拿不到评分逻辑。`Outcome.verifier` 只记录"谁验证的"这个标签，不包含验证逻辑本身。

**证据**：

- METR 实测：o3 在能看到评分函数时 30.4% 的 run 作弊，不可见时仅 0.7%——**43 倍差异**（[METR, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)）。
- Anthropic 进一步发现：生产 RL 中学会 reward hacking 的模型会泛化为广谱错位行为（[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)）。

### 红线 3：迭代熔断 —— `IterationController`

**规则**：`IterationController(max_rounds=3).record_round(metric)` 记录每一轮自训练及其评估指标；超过上限抛 `IterationHalted`，强制停止该方向的迭代。`QLoRADPOConfig.max_rounds=3` 是同一纪律在训练配置层的体现。

**证据**：

- Self-Rewarding LM（Meta）：3 轮迭代 DPO 后 AlpacaEval 胜率 9.94%→20.44%，但**第 4 轮后普遍饱和**（[arXiv:2401.10020](https://arxiv.org/abs/2401.10020)；DICE 复现确认）。
- ReST-EM：MATH +5.9%，但 APPS 上**第 2 轮迭代即过拟合回退**（[arXiv:2312.06585](https://arxiv.org/abs/2312.06585)）。
- 经验法则（由上述论文证据归纳）：自我迭代 2–3 轮后必须更换数据分布或停止；每轮之间用冻结评估集裁决。
- 若上线 RL 类方法，另需监控 RAGEN 命名的 **Echo Trap**（奖励方差塌缩 → 熵降 → 策略同质化，[arXiv:2504.20073](https://arxiv.org/abs/2504.20073)）。

---

## 7. 包结构与依赖策略

```
evoflow/
├── core/            # 零依赖：signals, trajectory, gate, discipline
├── llm/             # v0.2 新增：零依赖 OpenAI 兼容客户端 + LLMClient 协议 + ScriptedLLM 测试桩
├── l0_context/      # 零依赖：playbook, reflector；v0.2 新增 llm_reflector（ACELoop）、gepa
├── l1_memory/       # 零依赖：reasoning_bank, store；v0.2 新增 episodic、profile、qdrant_store（[vec-qdrant]）
├── l2_weight/       # flywheel/registry/serving 零依赖；trainers/ 需 [train]（v0.2 重写 qlora_dpo + 新增 echo_trap）
├── knowledge/       # v0.2 新增：VectorRAG / LightGraphRAG / 路由，零依赖
├── telemetry/       # v0.2 新增：进化事件流水 + 组合式插桩；OTel/Langfuse 导出需 [otel]
├── integrations/    # v0.2 新增：MCP Server（stdio 模式零依赖；SDK 增强模式需 [mcp]）
└── eval/            # 零依赖：golden_set, judge（LLMJudge 需 [llm]）；v0.2 新增 calibration、leakage
```

- **核心包零第三方依赖**：中小企业可以 `pip install evoflow` 后直接跑 L0/L1/评估门全闭环，不需要 GPU、不需要向量数据库、不需要嵌入模型。
- **`evoflow[train]`**：torch / peft / trl / bitsandbytes 等训练栈，只在 L2 训练时需要。
- **`evoflow[vec]`**：numpy，`VectorStore` Protocol 的数值向量运算支持。
- **`evoflow[vec-qdrant]`**（v0.2 新增）：qdrant-client，`QdrantStore` 生产向量后端（惰性 import，核心包零依赖不变）。
- **`evoflow[llm]`**：OpenAI 兼容客户端，用于把 LLM 接入 `reflect_fn` 和 `LLMJudge`（可接国产低价 API，见 [integration.md](integration.md)）。
- **`evoflow[otel]`**（v0.2 新增）：opentelemetry-sdk / langfuse，`evoflow.telemetry` 事件流水的增量导出。
- **`evoflow[mcp]`**（v0.2 新增）：官方 mcp 包，MCP Server 的 FastMCP 增强模式；默认 stdio 模式零依赖，不装也能用。

这条依赖边界本身就是架构决策：**进化逻辑（框架）与模型能力（注入的函数）解耦**——reflect_fn、embed_fn、judge 都是可替换的注入点。

---

## 8. 已知局限（诚实声明）

1. 各层收益数字（ACE +10.6%、ReasoningBank +8.3pp 等）均为论文自测，**你的场景必须用自有黄金集复测**——这正是评估门存在的意义。
2. 部分工程成本数字（QLoRA 单次 <¥2、50 条 DPO +30%）来自未审计的工程博客，仅作量级参考。
3. `JsonlVectorStore` 的 char-3gram hashing 向量是零依赖的取舍：语义检索能力弱于真正的嵌入模型，记忆库规模上千条后建议接入外部嵌入（`evoflow[vec]`）。
4. 评估质量决定整个系统的上限：黄金集太弱，评估门形同虚设。先建 50–100 条金标，再谈进化。
5. **（v0.2）L2 训练主流程未经 GPU 实测**：`qlora_dpo.py` 重写后的配置校验、数据校验与熔断回调有单测覆盖，但真实训练行为（显存占用、收敛、速度）请先在自有硬件上小规模验证；`LightGraphRAG` 是教学级简化实现，生产环境建议替换为真 LightRAG / LazyGraphRAG。

---

## 参考

- ACE — [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- ReasoningBank — [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
- Dynamic Cheatsheet — [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)
- Self-Rewarding LM — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
- ReST-EM — [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
- METR reward hacking — [metr.org, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
- 自进化五件工程综述 — [AgentsPulse, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/)
