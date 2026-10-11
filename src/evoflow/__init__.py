"""EvoFlow —— 三层进化栈的自进化 Agent 框架。

架构分层（对应公开研究：ACE / ReasoningBank / QLoRA 等）：
- L0 上下文进化（类 ACE）：Generator -> Reflector -> Curator，playbook 增量 delta 条目；
  v0.2 新增 LLMReflector/ACELoop 开箱闭环与 GEPA 提示词进化器。
- L1 经验/记忆库（类 ReasoningBank）：从成功+失败轨迹蒸馏策略条目，嵌入检索，k=1 注入；
  v0.2 新增 A-MEM 式情节记忆、用户画像记忆、Qdrant 向量后端。
- L2 权重层（数据飞轮）：反馈信号 -> 偏好对 -> QLoRA+DPO 训练；adapter 版本化注册表，
  可晋升/回滚；v0.2 新增生产级训练后端、Echo Trap 监控与 vLLM 热切换。
- 知识层（v0.2 新增）：知识放在参数之外——VectorRAG（混合检索+引用溯源）、
  LightGraphRAG（双层检索）、长上下文/向量/图谱路由建议。
- 贯穿层：评估门（冻结黄金集回归测试，不过线不晋升）、反馈信号采集、迭代熔断；
  v0.2 新增进化可观测性（EvolutionHistory + OTel/Langfuse 导出）、评委校准与黄金集防泄漏。
- 集成层（v0.2 新增）：MCP Server，任何 MCP 宿主可直接获得进化能力。

红线（硬约束）：
1. 评估门不过线禁止晋升（见 :class:`AdapterRegistry.promote`）。
2. judge/评分器与 agent 运行时隔离（见 :class:`evoflow.eval.judge.LLMJudge`）。
3. 迭代超过 max_rounds 抛 :class:`IterationHalted`（见 :class:`IterationController`）。
4. 黄金集不得进入训练数据（见 :class:`evoflow.eval.leakage.assert_no_leakage`）。
"""

from evoflow.core.discipline import IterationController, IterationHalted
from evoflow.core.gate import EvolutionGate, GateDecision, GateResult
from evoflow.core.signals import FeedbackSignal, SignalCollector, SignalType
from evoflow.core.trajectory import Outcome, Trajectory
from evoflow.eval.calibration import CalibrationReport, calibrate, cohens_kappa
from evoflow.eval.golden_set import GoldenCase, GoldenSet
from evoflow.eval.judge import Judge, LLMJudge, RuleBasedJudge
from evoflow.eval.leakage import (
    LeakageError,
    LeakageReport,
    assert_no_leakage,
    check_leakage,
)
from evoflow.knowledge.base import Chunk, Document, KnowledgeLayer
from evoflow.knowledge.graph_rag import LightGraphRAG
from evoflow.knowledge.router import recommend as recommend_knowledge_layer
from evoflow.knowledge.vector_rag import VectorRAG
from evoflow.l0_context.gepa import GEPAEvolver, PromptCandidate, pareto_front
from evoflow.l0_context.llm_reflector import ACELoop, LLMReflector
from evoflow.l0_context.playbook import DeltaEntry, Playbook
from evoflow.l0_context.reflector import Reflector
from evoflow.l1_memory.episodic import EpisodicMemory, MemoryNote
from evoflow.l1_memory.profile import ProfileMemory, UserProfile
from evoflow.l1_memory.reasoning_bank import ReasoningBank, StrategyItem
from evoflow.l1_memory.store import JsonlVectorStore, VectorStore
from evoflow.l2_weight.flywheel import DataFlywheel, PreferencePair
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied
from evoflow.llm.client import LLMClient, LLMError, OpenAICompatClient, ScriptedLLM
from evoflow.telemetry.events import EventKind, EvolutionEvent
from evoflow.telemetry.history import EvolutionHistory, wrap_gate, wrap_registry

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "FeedbackSignal",
    "SignalCollector",
    "SignalType",
    "Outcome",
    "Trajectory",
    "EvolutionGate",
    "GateDecision",
    "GateResult",
    "IterationController",
    "IterationHalted",
    # llm
    "LLMClient",
    "LLMError",
    "OpenAICompatClient",
    "ScriptedLLM",
    # L0
    "DeltaEntry",
    "Playbook",
    "Reflector",
    "LLMReflector",
    "ACELoop",
    "GEPAEvolver",
    "PromptCandidate",
    "pareto_front",
    # L1
    "ReasoningBank",
    "StrategyItem",
    "VectorStore",
    "JsonlVectorStore",
    "EpisodicMemory",
    "MemoryNote",
    "ProfileMemory",
    "UserProfile",
    # L2
    "DataFlywheel",
    "PreferencePair",
    "AdapterRegistry",
    "PromotionDenied",
    # eval
    "GoldenCase",
    "GoldenSet",
    "Judge",
    "LLMJudge",
    "RuleBasedJudge",
    "CalibrationReport",
    "calibrate",
    "cohens_kappa",
    "LeakageError",
    "LeakageReport",
    "assert_no_leakage",
    "check_leakage",
    # knowledge
    "Chunk",
    "Document",
    "KnowledgeLayer",
    "VectorRAG",
    "LightGraphRAG",
    "recommend_knowledge_layer",
    # telemetry
    "EvolutionEvent",
    "EventKind",
    "EvolutionHistory",
    "wrap_gate",
    "wrap_registry",
]
