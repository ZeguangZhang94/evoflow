# 贡献指南（Contributing to EvoFlow）

感谢你对 EvoFlow 的兴趣！在开始前，请先理解本项目的一条元规则：

> **EvoFlow 是研究驱动的框架：每个类、每个默认参数、每条硬约束都要能溯源到一条研究结论**（见 [docs/research-to-code.md](docs/research-to-code.md)）。贡献代码前请先回答："这个改动对应哪条研究结论或真实工程证据？"

---

## 目录

1. [开发环境搭建](#1-开发环境搭建)
2. [代码风格](#2-代码风格)
3. [如何跑测试](#3-如何跑测试)
4. [PR 流程](#4-pr-流程)
5. [核心设计约束（贡献者必读）](#5-核心设计约束贡献者必读)
6. [Good First Issue 方向](#6-good-first-issue-方向)

---

## 1. 开发环境搭建

```bash
git clone https://github.com/<org>/evoflow.git
cd evoflow

# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 开发安装（含核心 + 测试工具）
pip install -e ".[dev]"

# 如需开发训练/向量/LLM 相关功能，加装对应 extras
pip install -e ".[dev,train,vec,llm]"
```

要求 Python ≥ 3.10。**核心包本身零第三方依赖**——日常大部分开发（core / l0 / l1 / eval）不需要装任何 extras。

## 2. 代码风格

- **格式化与 lint**：`ruff`（格式化 + lint 一把梭）。提交前运行：

  ```bash
  ruff format .
  ruff check .
  ```

- **类型标注**：所有公开 API 必须有完整 type hints，CI 跑 `mypy evoflow/`。
- **命名与结构**：
  - 公开 API 只通过各模块顶层导出（如 `from evoflow.core.gate import EvolutionGate`），不要暴露内部实现细节。
  - dataclass 优先于 dict 传参（如 `FeedbackSignal`、`GateResult`、`DeltaEntry`）。
- **文档字符串**：公开类/方法用中文写 docstring，说明**它对应哪条研究结论**（附 arXiv 编号或公开来源链接）。示例：

  ```python
  def retrieve(self, query: str, k: int = 1) -> list[StrategyItem]:
      """检索最相关的 k 条策略注入 prompt。

      k=1 是 ReasoningBank 论文（arXiv:2509.25140）的最优设置：
      相关性 > 数量。除非有复测证据，不建议调大。
      """
  ```

- **默认参数**：修改任何公开默认参数（如 `min_pass_rate`、`pos_neg_ratio`、`k=1`）视同行为变更，PR 中必须说明研究/工程依据，并更新对应文档。

## 3. 如何跑测试

```bash
# 核心测试（零依赖，任何环境可跑）——这是你 PR 必须全绿的部分
pytest tests/ -m "not train and not vec and not llm"

# 训练相关（需 evoflow[train]，用小模型 CPU 路径）
pytest tests/ -m train

# 向量库相关（需 evoflow[vec]）
pytest tests/ -m vec

# LLM 相关（需 evoflow[llm]，外部 API 用 mock，不打真实请求）
pytest tests/ -m llm

# 全量
pytest tests/
```

测试约定：

- **不允许在测试中打真实外部 API**：LLM/嵌入一律 mock 或注入假函数。
- 核心逻辑的测试要覆盖"硬约束"行为：评估门不过线时 `passed=False`、`IterationController` 超限抛 `IterationHalted`、`k` 默认值等。
- 新增 bug 修复必须附带复现该 bug 的失败测试（先红后绿）。

## 4. PR 流程

1. **先开 issue 讨论**（小修复除外）：尤其是新功能——issue 模板会要求你说明"对应哪条研究结论"，讨论通过再动手，避免白做。
2. Fork + 特性分支：`feat/xxx`、`fix/xxx`、`docs/xxx`。
3. 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：`feat: ...` / `fix: ...` / `docs: ...` / `refactor: ...`。
4. PR 模板检查清单逐项打勾，重点三条：
   - 核心包未引入新的第三方依赖；
   - 公开 API 变更已同步更新 README / docs 中的示例（**文档里的代码就是 API 契约**）；
   - CHANGELOG.md 的 Unreleased 段已更新。
5. CI 全绿（lint / mypy / 4 个 Python 版本的裸环境核心测试 / extras 测试 / 文档示例检查）。
6. 至少一名维护者 review。review 重点除代码质量外，还包括：**证据标注是否完整**（引用的数字是否标注了"论文自测/未审计"等审计状态）。
7. Squash merge 到 main。

## 5. 核心设计约束（贡献者必读）

违反以下任一条的 PR 将被直接拒绝：

1. **核心包零第三方依赖**。`core/`、`l0_context/`、`l1_memory/`、`eval/`、`l2_weight/` 的 flywheel 与 registry 部分，只能 import 标准库。重依赖一律进 extras 并在 import 时延迟加载、给出友好报错（如 `pip install evoflow[train]`）。
2. **评分器隔离**：`evoflow.eval` 不得被 `l0_context` / `l1_memory` / 任何 agent 运行路径 import。judge、黄金集、验证逻辑永远只在评估侧。（依据：METR 43 倍 reward hacking 差异。）
3. **硬约束不可绕过**：`IterationController` 超限必须抛 `IterationHalted`（异常，不是 warning）；`EvolutionGate` 的 `min_pass_rate` 不允许有"宽松模式"快捷开关。
4. **黄金集不进训练数据**：任何把 `GoldenSet` 内容写入训练数据/记忆库的代码路径都是红线。
5. **数字引用带标注**：文档和 docstring 中引用性能/成本数字时，必须保留来源与审计状态标注（论文自测 / 第三方复现 / 未审计工程文）。夸大宣传是本项目最大的信誉风险。

## 6. Good First Issue 方向

适合新贡献者的切入点（大致按难度排序）：

- **文档**：翻译、错别字、示例代码补充；为 docstring 补研究依据引用。
- **RuleBasedJudge 的规则集**：为 `evoflow.eval.judge` 增加更多开箱即用的规则判分器（精确匹配、包含、正则、JSON schema 校验等）。纯标准库，零依赖。
- **JsonlVectorStore 改进**：并发写安全、向量归一化优化、检索性能基准脚本。
- **`evoflow init` 模板完善**：更丰富的新项目骨架（示例 golden.jsonl、示例 reflect_fn）。
- **examples/**：端到端示例（客服场景、报价场景），只用核心包 + mock LLM。
- **更多 VectorStore 实现**：SQLite-vec、Chroma 等（进 `evoflow[vec]`）。
- **研究证据勘误**：发现文档/代码中引用的数字有更新或复现失败？开 `evidence` 标签的 issue——这类贡献与代码同等重要。

不确定从哪里开始？在 GitHub Discussions 打个招呼，说明你的背景和兴趣，维护者会帮你找一个合适的切入点。

---

再次感谢你的时间与兴趣 —— EvoFlow 的目标是成为"中小企业敢用在生产上的自进化框架"，这需要每一个 PR 的严谨。🙌
