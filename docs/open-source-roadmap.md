# EvoFlow 开源发布与运营路线图

> 面向维护者的内部路线文档：仓库初始化、PyPI 发布、CI、版本策略、社区冷启动、里程碑规划。
> 原则：**先把"研究结论 → 代码"的差异化讲清楚，再谈 star 数。** EvoFlow 的卖点不是"又一个 agent 框架"，而是"每个设计决策都可溯源到论文证据"（见 [research-to-code.md](research-to-code.md)）。

---

## 1. 仓库初始化 Checklist

发布前逐项打勾：

- [ ] `LICENSE`：Apache-2.0 全文（与 README 声明一致；Apache-2.0 与 Qwen3 等主流开源基座许可兼容，对商业使用友好）
- [ ] `README.md`：中英双语、架构图、5 分钟快速开始、徽章（已有）
- [ ] `CONTRIBUTING.md`：代码风格、测试、PR 流程、good first issue（已有）
- [ ] `CODE_OF_CONDUCT.md`：采用 [Contributor Covenant v2.1](https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/)，中英双语
- [ ] `SECURITY.md`：漏洞报告渠道（安全邮箱，而非公开 issue）
- [ ] `CHANGELOG.md`：遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式
- [ ] `.github/ISSUE_TEMPLATE/`：bug 报告 / 功能请求 / 研究证据勘误（见 §5）
- [ ] `.github/PULL_REQUEST_TEMPLATE.md`（见 §5）
- [ ] `.github/workflows/`：CI（见 §4）
- [ ] `.gitignore`：Python 标准模板 + `playbook.json` / `memory.jsonl` / `adapters/` / `*.gguf` 等运行时产物
- [ ] `pyproject.toml`：核心零依赖声明 + extras（`train` / `vec` / `llm`）+ CLI entry point
- [ ] `docs/`：architecture / research-to-code / integration / 本文件（已有）
- [ ] GitHub 仓库设置：开 Issues / Discussions，关 Wiki（文档都在仓库内），设置 topics：`llm-agents` `self-evolving` `qlora` `dpo` `ace` `reasoningbank` `sme`
- [ ] 仓库描述一句话：`Self-evolving LLM agents for SMBs: three-layer evolution stack (ACE-style playbook + ReasoningBank + QLoRA/DPO flywheel) behind a frozen evaluation gate.`

## 2. PyPI 发布步骤

> 前置确认：`evoflow` 包名在 PyPI 未被占用（发布前务必检查 `https://pypi.org/project/evoflow/`，被占则备选 `evoflow-ai` / `evo-flow`）。

```bash
# 1. 版本号写入 pyproject.toml（如 0.1.0），更新 CHANGELOG.md
# 2. 构建
python -m pip install build twine
python -m build

# 3. 先在 TestPyPI 验证
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ evoflow
python -c "import evoflow; print(evoflow.__version__)"

# 4. 正式发布
twine upload dist/*

# 5. 验证 extras 解析正确
pip install "evoflow[train]"   # 应拉入 torch/peft/trl/bitsandbytes
pip install "evoflow[vec]"     # 应拉入 qdrant-client
pip install "evoflow[llm]"     # 应拉入 openai

# 6. 打 git tag 并推送
git tag v0.1.0 && git push origin v0.1.0

# 7. 在 GitHub 创建 Release，附 CHANGELOG 对应段落
```

发布纪律：

- **tag、PyPI 版本、CHANGELOG、GitHub Release 四处版本号必须一致**。
- 用 [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)（OIDC）替代长期 token，发布动作由 GitHub Actions 在 tag 推送时自动执行。
- 核心包零依赖是硬性承诺：CI 必须有"裸环境 import 测试"（见 §4），防止有人把重依赖写进核心包。

## 3. 语义化版本策略

遵循 [SemVer 2.0](https://semver.org/lang/zh-CN/)，结合本项目特点细化：

| 版本段 | 变更内容 | 示例 |
|---|---|---|
| MAJOR（x.0.0） | API 契约破坏性变更：公开类/函数签名变化、`GateResult` 等 dataclass 字段删除、CLI 子命令改名 | `Trajectory` 移除 `steps` 字段 |
| MINOR（0.x.0） | 向后兼容的新功能：新模块、新 extras、新 CLI 子命令、新 VectorStore/Judge 实现 | 新增 GraphRAG 适配器 |
| PATCH（0.0.x） | 向后兼容的修复：bug fix、文档、性能优化、默认参数内部实现改进（不改签名） | 修复 JsonlVectorStore 并发写 |

0.x 阶段的特殊约定：

- **v0.x 允许 MINOR 级破坏**，但每次必须在 CHANGELOG 的 `Breaking` 小节明确列出，并给出迁移说明。
- **API 契约冻结线**：`evoflow.core`（signals / trajectory / gate / discipline）从 v0.2 起视为准冻结——这四个模块是用户业务流程的锚点，破坏它们等于破坏用户的反馈数据格式。
- **默认参数变更视为 MINOR 级**：如 `min_pass_rate` 默认值改动会改变用户系统行为，必须进 CHANGELOG 并说明研究依据。
- 发布节奏：PATCH 随时；MINOR 每 4–8 周；不追热点赶 release。

## 4. GitHub Actions CI 建议

`.github/workflows/ci.yml` 建议包含以下 job（按 PR 触发 + 主分支触发）：

```yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff && ruff check . && ruff format --check .
      - run: pip install mypy && mypy evoflow/

  test-core:   # 关键 job：证明核心包零依赖
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - run: pip install -e .          # 不装任何 extras
      - run: python -c "import evoflow.core.gate, evoflow.l0_context.playbook, evoflow.l1_memory.reasoning_bank, evoflow.eval.golden_set"
      - run: pip install pytest && pytest tests/ -m "not train and not vec and not llm"

  test-extras:  # 带 extras 的集成测试（可用 CPU-only torch）
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[train,vec,llm]" pytest
      - run: pytest tests/ -m "train or vec or llm" --maxfail=1

  docs-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/check_doc_examples.py   # 提取 docs 中的代码块做语法/导入检查
```

要点：

- **`test-core` 是本项目的生命线 job**：在**不装任何 extras** 的裸环境跑全部核心测试，任何把第三方依赖泄漏进核心包的 PR 会在这里挂掉。
- 文档示例必须有 `check_doc_examples.py` 之类的脚本兜底——文档里的 API 与实现漂移是开源项目最常见的失信来源。本项目 README 的 API 是契约（见 [research-to-code.md](research-to-code.md)），漂移 = 违约。
- 训练类测试用小模型（如 `Qwen3-0.6B`）+ CPU/小显存路径，单次 CI 控制在 10 分钟内。

## 5. Issue / PR 模板建议

### `.github/ISSUE_TEMPLATE/bug_report.yml`

关键字段：EvoFlow 版本、Python 版本、安装方式（是否含 extras）、最小复现（必须包含 trajectory/signal 的构造代码）、期望 vs 实际行为。**特别要求注明**：问题发生在哪一层（L0/L1/L2/评估门/CLI）——与架构文档对齐，方便分流。

### `.github/ISSUE_TEMPLATE/feature_request.yml`

关键字段：

- **这个功能对应哪条研究结论？**（请给出论文/公开来源链接）——与 [research-to-code.md](research-to-code.md) 的准入原则一致，答不上来的功能请求默认降级到 Discussions 讨论。
- 它应该进核心包还是 extras？（判断标准：不引入第三方依赖才可能进核心包）

### `.github/ISSUE_TEMPLATE/evidence_correction.yml`（本项目特色）

**研究证据勘误**：本项目大量设计基于具体论文数字。模板字段：涉及的文档/代码位置、原数字与来源、新证据（论文更新/复现失败/勘误）、建议的修正。这类 issue 打 `evidence` 标签，优先级高于普通文档 issue——**证据的准确性就是本项目的信誉本身**。

### `.github/PULL_REQUEST_TEMPLATE.md`

```markdown
## 变更内容
## 对应 issue / 研究依据
## 检查清单
- [ ] 核心包未引入新的第三方依赖（如引入，请说明为何不能放 extras）
- [ ] 新增/修改的公开 API 已同步更新文档示例（README / docs/）
- [ ] 测试覆盖：核心逻辑有单测；训练/向量/LLM 相关已打对应 pytest mark
- [ ] CHANGELOG.md 已更新（Unreleased 段）
- [ ] 若涉及默认参数变更：已说明研究依据并标注 Breaking/Behavior-change
```

## 6. 社区冷启动建议

### 目标社区与切入角度

| 社区 | 语言 | 切入角度 | 形式 |
|---|---|---|---|
| **知乎** | 中 | 主打低成本自进化的落地叙事：月成本千元级、QLoRA 单次 <¥10、评估门防退化。所有数字标注公开来源，写长文而非广告 | 专栏长文（技术解读 + 框架发布） |
| **机器之心 / 量子位投稿** | 中 | 主打"ACE + ReasoningBank + QLoRA 飞轮的工程整合"，蹭 ICLR 2026 两篇论文热度 | 投稿技术解读稿 |
| **Reddit r/LocalLLaMA** | 英 | 主打 zero-dependency core + $1–10 QLoRA iteration + "your agent gets worse without a gate"（METR 43× 数字很有传播力） | Show 帖：标题建议 "EvoFlow: self-evolving agents behind a frozen eval gate (ACE + ReasoningBank + QLoRA/DPO flywheel)" |
| **HackerNews** | 英 | 主打研究驱动定位："Every default parameter cites a paper"。HN 吃"诚实声明"这一套——把厂商自测标注、已知局限放在显眼处反而是加分项 | Show HN（选周二至周四美东上午发） |
| **X / Twitter** | 英 | 线程讲三层栈 ASCII 图 + 三个关键数字 | 图文线程，@相关论文作者 |
| **GitHub 生态** | — | 在 awesome-llm-agents / awesome-ai-agents 等列表提 PR；给 ACE / ReasoningBank 官方仓库的 "third-party implementations" 区提 PR | PR |

### 冷启动纪律

1. **先发文档质量，再发代码完成度**：v0.1 哪怕功能少，README + 三篇 docs 必须是完整体——研究驱动定位要求文档即产品。
2. **不要在宣发材料里裸用论文头条数字**：必须带"论文自测"标注（本项目文档已全线遵守）。被 r/LocalLLaMA 抓到夸大宣传是灾难性的。
3. **第一波 seed 用户画像**：正在用 Dify/Coze 搭客服/销售 agent、遇到"上线后不会变好"痛点的中小团队。去 Dify/FastGPT 的中文社区（微信群、论坛）讲"怎么让 Dify 应用具备经验库和评估门"（对接 [integration.md](integration.md)）。
4. **响应 SLA**：发布首月 issue 24 小时内首响；首月收到的前 10 个真实使用反馈，逐条进 roadmap 讨论。
5. **指标北极星**：不是 star，而是"有真实业务反馈信号跑通完整闭环（distill → gate → promote）的外部团队数"。

## 7. 里程碑规划

### v0.1 —— 核心闭环（当前）

> 目标：零依赖跑通 L0 + L1 + 评估门，证明"免训练自进化闭环"可用。

- `core`：signals / trajectory / gate / discipline 四个模块
- `l0_context`：Playbook + Reflector + DeltaEntry
- `l1_memory`：ReasoningBank + VectorStore Protocol + JsonlVectorStore
- `eval`：GoldenSet + RuleBasedJudge + LLMJudge（Protocol 先行）
- `l2_weight`：DataFlywheel + AdapterRegistry（flywheel 数据侧先行，训练器留接口）
- CLI 五子命令；全套文档（README/architecture/research-to-code/integration）
- **发布出口条件**：核心测试在 4 个 Python 版本裸环境全绿；README 快速开始示例可运行。

### v0.2 —— 训练后端

> 目标：数据飞轮闭环到权重层，QLoRA + DPO 一键训练 + adapter 生命周期管理。

- `l2_weight.trainers.qlora_dpo` 完整实现（`evoflow[train]`）
- `AdapterRegistry.promote/rollback` 与 `EvolutionGate` 的联动示例（examples/ 目录）
- `IterationController` 与 `QLoRADPOConfig.max_rounds` 的双重熔断集成测试
- 训练成本实测报告：在 RTX 4090 / 云租 GPU 上跑 Qwen3-8B，记录单次迭代真实成本，写进文档（用自有实测替换引用数字）
- **发布出口条件**：从 `FeedbackSignal` 到 `promote` 的端到端 example 可运行。

### v0.3 —— 生态集成

> 目标：进入用户已有的 agent 技术栈，而不是要求用户换栈。

- `evoflow[vec]`：Qdrant 版 VectorStore 实现
- LangGraph / CrewAI / Dify 集成适配器与示例（按 [integration.md](integration.md) 固化成代码）
- RAG / GraphRAG（LightRAG）集成示例：经验库 + 知识库联合注入
- OpenAI 兼容 API 的 `reflect_fn` / `LLMJudge` 官方实现（含国产低价 API 配置示例）
- **发布出口条件**：每个集成目标有可运行 example + 文档截图/录屏。

### v1.0 —— 生产级

> 目标：API 冻结，敢让用户押注。

- `evoflow.core` API 正式冻结（SemVer 全约束生效）
- 完整性能基准：L0/L1 在自有基准上的收益复测报告（诚实标注与论文数字的差异）
- 多后端评测：不同基座（Qwen3-8B/14B、GLM-4.5-Air 等）上的闭环效果对比
- 文档站（mkdocs-material 或类似）、API reference 全覆盖
- 至少 3 个外部团队的真实闭环案例（北极星指标）

### 里程碑之外（明确不做 / 缓做）

- **不做自己的 agent 运行时**：EvoFlow 是进化层，嵌入别人的运行时（LangGraph 等），不与之竞争。
- **缓做多智能体架构进化**：CMU 综述的 What 维度之一，但中小企业场景证据不足，等社区需求信号。
- **不做托管 SaaS**：开源信任是冷启动资产，托管化至少等 v1.0 之后。
