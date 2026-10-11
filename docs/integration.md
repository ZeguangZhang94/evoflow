# EvoFlow 生态集成指南

> EvoFlow 是**进化层**，不是 agent 运行时——它嵌入你已有的技术栈（LangGraph / CrewAI / Dify），让现有应用获得"自己变好"的能力，而不是要求你换栈。
>
> 本文给出三类集成：① 在主流 agent 框架中嵌入三层进化栈；② 接 Qdrant 做 L1 向量后端；③ 接 OpenAI 兼容 API（含国产低价 API）做 `reflect_fn` 和 `LLMJudge`。

---

## 0. 集成的统一心智模型

无论嵌进哪个框架，EvoFlow 的挂载点永远是四个：

```
任务执行前：  playbook.render(max_tokens) + bank.retrieve(query, k=1)  → 注入 prompt
任务执行后：  可验证结果 → Trajectory / FeedbackSignal 落库
离线/定期：   reflect → curate / distill；build_pairs → train（可选）
发布前：     EvolutionGate.evaluate(candidate) → promote / rollback
```

记住红线：**评估侧（黄金集、judge）只在离线流水线中，运行中的 agent 永远接触不到**（METR 43 倍 reward hacking 差异，见 [architecture.md](architecture.md) §6）。

---

## 1. 在 LangGraph 中嵌入

思路：用 LangGraph 的节点包装 EvoFlow 的三个挂载点，graph 结构不变，只加"进化前置节点"和"反馈收集节点"。

```python
from langgraph.graph import StateGraph, END
from evoflow.l0_context.playbook import Playbook
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore
from evoflow.core.trajectory import Trajectory, Outcome
from evoflow.core.signals import FeedbackSignal, SignalType, SignalCollector

playbook = Playbook().load("playbook.json")
bank = ReasoningBank(store=JsonlVectorStore("memory.jsonl"), embed_fn=None)
collector = SignalCollector()

# ① 进化前置节点：注入 playbook + 检索策略
def evolve_node(state: dict) -> dict:
    context = playbook.render(max_tokens=800)
    strategies = bank.retrieve(state["task"], k=1)
    strategy_text = "\n".join(s.content for s in strategies)
    state["system_prompt"] = f"{context}\n\n# 相关经验\n{strategy_text}"
    return state

# ② 你的原有业务节点（LLM 调用、工具调用……），只是把 prompt 换成 state["system_prompt"]
def business_node(state: dict) -> dict:
    # response = your_llm(system=state["system_prompt"], user=state["task"])
    state["result"] = "..."          # 你的业务逻辑
    state["steps"] = ["..."]          # 记录执行轨迹
    return state

# ③ 反馈收集节点：把可验证结果落成 Trajectory + Signal
def feedback_node(state: dict) -> dict:
    success = verify(state["result"])          # 你的可验证逻辑：规则/单测/用户标记
    traj = Trajectory(task=state["task"], steps=state["steps"],
                      outcome=Outcome(success=success, verifier="rule", detail=""))
    bank.distill(traj)                          # L1：成败都蒸馏
    collector.add(FeedbackSignal(
        type=SignalType.SUCCESS if success else SignalType.FAIL,
        task_id=state["task_id"], value=1.0 if success else 0.0, meta={}))
    return state

graph = StateGraph(dict)
graph.add_node("evolve", evolve_node)
graph.add_node("business", business_node)
graph.add_node("feedback", feedback_node)
graph.set_entry_point("evolve")
graph.add_edge("evolve", "business")
graph.add_edge("business", "feedback")
graph.add_edge("feedback", END)
app = graph.compile()
```

L0 的 `Reflector.reflect()` 建议**离线批量跑**（比如每天一次对当日轨迹反思后 `playbook.curate()`），而不是每请求实时反思——控制 LLM 调用成本，也便于评估门批量验证 playbook 变更。

## 2. 在 CrewAI 中嵌入

思路：CrewAI 的 Agent 定义支持注入 `system_template` / 通过 task callback 收集结果，把 EvoFlow 挂在 prompt 模板与回调上。

```python
from crewai import Agent, Task, Crew
from evoflow.l0_context.playbook import Playbook
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore

playbook = Playbook().load("playbook.json")
bank = ReasoningBank(store=JsonlVectorStore("memory.jsonl"), embed_fn=None)

def make_agent(role: str, goal: str, task_query: str) -> Agent:
    strategies = bank.retrieve(task_query, k=1)
    evolved_backstory = (
        playbook.render(max_tokens=600)
        + "\n# 相关经验\n"
        + "\n".join(s.content for s in strategies)
    )
    return Agent(role=role, goal=goal, backstory=evolved_backstory)

def on_task_done(output):
    # 任务回调：把结果可验证化后落库（同 LangGraph 的 feedback_node）
    ...

agent = make_agent("客服专员", "处理客户投诉", task_query="客户投诉退款")
task = Task(description="...", agent=agent, callback=on_task_done)
crew = Crew(agents=[agent], tasks=[task])
crew.kickoff()
```

要点：CrewAI 多 agent 场景下，**每个角色维护独立 playbook 文件**（`playbook_sales.json` / `playbook_support.json`），避免不同角色的经验互相污染；评估门也按角色各建一个黄金集。

## 3. 在 Dify 中嵌入

Dify 是低代码平台，不能改源码时走两条路：

### 路线 A：Dify 外部包一层（推荐，侵入最小）

把 Dify 应用当"LLM 执行器"，EvoFlow 跑在外层 Python 服务里：

```python
# 你的 FastAPI 服务（客户系统 → 本服务 → Dify API）
from evoflow.l0_context.playbook import Playbook

playbook = Playbook().load("playbook.json")

def handle_user_query(query: str):
    # ① 调 Dify 前，把 playbook + 策略拼进 query 的 context 字段
    context = playbook.render(max_tokens=800)
    resp = dify_client.chat(query=query, inputs={"evolved_context": context})
    # ② 用户侧反馈（点赞/点踩/人工标记）回流
    #    Dify 的 message feedback API → 转成 FeedbackSignal 落库
    ...
```

Dify 应用内只需在 prompt 模板加一个 `{{evolved_context}}` 变量。评估门、飞轮、训练全部在这个外层服务离线跑。

### 路线 B：Dify 自定义工具 / 代码节点

- 用 Dify 的**代码执行节点**内嵌一个轻量 playbook 读取逻辑（playbook 放外部存储，节点里只做 `render` 的字符串拼接）。
- 用**自定义工具**封装反馈落库 API，在工作流末尾调用。

⚠️ Dify 代码节点有沙箱限制（无文件系统持久化），playbook 读写必须走外部 HTTP 存储，不要在节点里依赖本地文件。

### 无论哪条路线

Dify 的点赞/点踩是现成的 `SignalType.ACCEPT/REJECT` 来源——对接它的 message feedback API 即可完成"反馈信号架构先行"的落地。

## 4. 接 Qdrant（L1 向量后端升级）

默认 `JsonlVectorStore`（char-3gram hashing）适合零依赖起步；记忆库规模上千条、需要语义检索时，实现 `VectorStore` Protocol 接 Qdrant（`pip install "evoflow[vec]"`，Qdrant 自托管 100 万向量约 $0.10/小时的量级，以官方实时定价为准）：

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from evoflow.l1_memory.reasoning_bank import ReasoningBank, StrategyItem

class QdrantVectorStore:
    """实现 evoflow.l1_memory.store.VectorStore Protocol。"""

    def __init__(self, url: str = "http://localhost:6333",
                 collection: str = "evoflow_memory", dim: int = 1024):
        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def add(self, item_id: str, vector: list[float], payload: dict) -> None:
        self.client.upsert(self.collection,
                           [PointStruct(id=item_id, vector=vector, payload=payload)])

    def search(self, vector: list[float], k: int) -> list[dict]:
        hits = self.client.query_points(self.collection, query=vector, limit=k)
        return [h.payload for h in hits.points]

    # ...delete / count 等按 Protocol 补全

# embed_fn 换成真嵌入模型（见下一节的 OpenAI 兼容 API）
store = QdrantVectorStore()
bank = ReasoningBank(store=store, embed_fn=my_embed_fn)
strategies = bank.retrieve("客户报价异议", k=1)   # k=1 保持不变
```

要点：

- `embed_fn` 是注入点：可以是任何 `(str) -> list[float]`——本地 bge-m3、API 嵌入都行，框架不绑定。
- 换后端**不需要改 ReasoningBank 的任何调用代码**，Protocol 即插即用。
- `StrategyItem.title` 建议作为嵌入的主要文本（检索锚点），`content` 放 payload 原样返回。

## 5. 接 OpenAI 兼容 API（`reflect_fn` 和 `LLMJudge`）

EvoFlow 不绑定任何模型供应商——`reflect_fn` 和 `LLMJudge` 都是注入点。任何 OpenAI 兼容端点（`pip install "evoflow[llm]"`）都可接入，包括国产低价 API（如 qwen-flash、deepseek-v3.2 等，价格为来源快照，以官方实时定价为准）。

### 5.1 统一客户端

```python
from openai import OpenAI

# 任何 OpenAI 兼容端点：改 base_url + api_key 即可
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼
    # base_url="https://api.deepseek.com/v1",                       # DeepSeek
    api_key="sk-xxx",
)
MODEL = "qwen-flash"   # 或 "deepseek-v3.2"、"gpt-4o-mini" …

def chat(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return resp.choices[0].message.content
```

### 5.2 LLM 版 `reflect_fn`

```python
import json
from evoflow.core.trajectory import Trajectory

REFLECT_PROMPT = """你是经验提炼器。阅读以下任务轨迹，提炼出可复用的增量经验条目。
轨迹：{traj}
结果：{outcome}
要求：只输出 JSON 数组，每项含 title / content / tags / helpful。
- 失败轨迹提炼"避坑"条目，成功轨迹提炼"做法"条目；
- 条目要具体可执行，不要正确的废话；
- 无新经验则输出 []。"""

def llm_reflect(traj: Trajectory):
    raw = chat(REFLECT_PROMPT.format(
        traj=traj.steps,
        outcome=f"success={traj.outcome.success}, detail={traj.outcome.detail}"))
    return json.loads(raw)   # 生产环境加 JSON 修复与字段校验

from evoflow.l0_context.reflector import Reflector
reflector = Reflector(reflect_fn=llm_reflect)
```

⚠️ 成本控制：反思是离线批量动作（每任务约 3 次 LLM 调用，ACE 论文口径）。建议按天批量处理当日轨迹，而非每请求实时反思。

### 5.3 `LLMJudge`

```python
from evoflow.eval.judge import Judge   # Protocol

class OpenAICompatibleJudge:
    """LLM-as-judge：语义判断 expected vs actual。需 evoflow[llm]。"""

    def score(self, input: str, expected: str, actual: str) -> float:
        raw = chat(f"""判断"实际输出"是否满足"期望输出"。只输出 PASS 或 FAIL。
输入：{input}
期望：{expected}
实际：{actual}""")
        return 1.0 if raw.strip().startswith("PASS") else 0.0

from evoflow.core.gate import EvolutionGate
from evoflow.eval.golden_set import GoldenSet

gate = EvolutionGate(golden_set=GoldenSet.from_jsonl("golden.jsonl"),
                     judge=OpenAICompatibleJudge(), min_pass_rate=0.9)
```

使用纪律：

1. **LLM-as-judge 与人工一致率约 85%**：上线前必须人工校准——抽 50 条人工标注，算 judge 与人工的一致率，不达标先换 prompt/换模型。
2. **成本控制**：judge 每次调用 $0.01–0.10；用分层抽样评估而非全量，把评测成本控制在业务负载 1:1 以内（有团队失控到 10 倍的案例）。
3. **能用规则就别用 LLM**：格式类、字段类断言用零成本的 `RuleBasedJudge`；语义类才上 `LLMJudge`。
4. **judge 模型 ≠ agent 模型**：评分与被评尽量用不同供应商/不同模型，降低同源偏见。

### 5.4 `embed_fn` 用同一端点

```python
def my_embed_fn(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-v4", input=text)
    return resp.data[0].embedding

bank = ReasoningBank(store=QdrantVectorStore(), embed_fn=my_embed_fn)
```

---

## 6. 集成检查清单

上线前过一遍：

- [ ] 业务流程有可验证结果（用户接受/拒绝、任务成败、单测）——没有反馈信号，自进化闭环是污染源（ACE 退化证据）
- [ ] playbook / 记忆库 / 黄金集 各有独立文件，且黄金集**不在**任何训练/检索数据路径上
- [ ] judge 与黄金集只在离线评估流水线，运行中 agent 不可见
- [ ] `IterationController` 接入训练流水线，`max_rounds` 未被调大
- [ ] playbook 变更、adapter 晋升都过了 `EvolutionGate`
- [ ] `AdapterRegistry` 的 rollback 路径演练过一次
- [ ] LLM judge 做过人工校准；反思/judge 的调用成本有预算上限

---

更多架构背景见 [architecture.md](architecture.md)；各集成方案对应的研究依据见 [research-to-code.md](research-to-code.md)。
