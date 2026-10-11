# EvoFlow

[简体中文](README.md) | **English**

> An LLM agent framework that improves itself — built for small and mid-sized businesses.
> A self-evolving LLM agent framework that gets better from your own feedback signals — at coffee-money cost.

<!-- badges: start -->
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/evoflow/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-github%20actions-lightgrey)](.github/workflows/)
<!-- badges: end -->

**EvoFlow is not "a better model" — it's "a system that gets better on its own"**: knowledge lives outside the parameters (RAG), experience lives outside the weights (playbook / experience bank), behavior is locked in at low cost with QLoRA, and every update passes through the Evolution Gate with one-command rollback.

---

## Why We Built EvoFlow

Most teams don't lack models — they lack a mechanism that lets an agent **keep improving from real usage**. EvoFlow's design stands on three publicly documented research facts:

| Key fact | What it means | Source |
|---|---|---|
| **A single QLoRA fine-tune costs $1–10** | When you really do need to touch the weights, the cost is already down to "a cup of coffee" territory (7B-class model, single rented RTX 4090) | [Spheron, 2026-03](https://www.spheron.network/blog/how-to-fine-tune-llm-2026/) |
| **ACE: +10.6%, training-free** | Maintaining nothing but an incrementally evolving context playbook — with zero weight changes — yields +10.6% task success on AppWorld and −86.9% adaptation latency | [arXiv:2510.04618](https://arxiv.org/abs/2510.04618) |
| **The five engineering pieces of an Evolution Gate** | The self-evolving-agent community consensus: collect trajectories → evaluate on a frozen holdout → propose single-layer updates → regression-test → versioned snapshots + rollback. All five are mandatory. | [AgentsPulse review, 2026-07](https://agentspulse.github.io/tutorials/self-evolving-agents-review-en/) |

In one sentence: **training-free evolution (L0/L1) captures the first wave of gains; low-cost training (L2) hardens repeatedly verified, stable behaviors; the Evolution Gate guarantees "only better, never worse."**

> ⚠️ Honest note: figures like +10.6% are the paper authors' self-reported numbers on public benchmarks; whether they reproduce in your business scenario depends on your feedback-signal quality and base-model strength. EvoFlow's built-in Evolution Gate exists precisely so you can validate gains on **your own golden set** instead of trusting someone else's headline numbers.

---

## Architecture: The Three-Layer Evolution Stack + Evolution Gate

```
┌──────────────────────────────────────────────────────────────┐
│  EvolutionGate — frozen golden set + regression tests;       │
│  nothing below the bar is promoted. Scorers / golden set     │
│  are invisible to the agent runtime (anti reward hacking)    │
└──────────────▲────────────────────────────▲──────────────────┘
               │ discard if below           │ promote if passed
   ┌───────────┴───────────┐   ┌────────────┴──────────────┐
   │ L0 Context Evolution  │   │ L1 Experience / Memory    │
   │ (real-time)           │   │ Bank (task-level)         │
   │ ACE-style playbook:   │   │ ReasoningBank-style:      │
   │ incremental delta     │   │ distill success+failure   │
   │ curation              │   │ trajectories, k=1 recall  │
   │ cost: LLM calls only  │   │ cost: ~$0                 │
   └───────────▲───────────┘   └────────────▲──────────────┘
               │ stable patterns harden     │ distilled to data pool
   ┌───────────┴────────────────────────────┴──────────────┐
   │ L2 Weight-Layer Data Flywheel (weekly cadence)        │
   │ feedback -> preference pairs -> QLoRA + DPO           │
   │ (~$1-10 per run); versioned adapters: promote /       │
   │ rollback; base model stays frozen forever             │
   └───────────────────────────────────────────────────────┘
   ┌───────────────────────────────────────────────────────┐
   │ Knowledge Layer: VectorRAG / LightGraphRAG / routing  │
   │ — knowledge lives outside the parameters: route by    │
   │ scale across long-context / vector / graph            │
   └───────────────────────────────────────────────────────┘
```

Design red lines (each backed by public research evidence — see [docs/architecture.md](docs/architecture.md)):

1. **The Evolution Gate is a hard constraint**: any layer update that fails the golden set (default pass_rate ≥ 0.9) is discarded.
2. **Scorer isolation**: scoring functions, verifiers, and the golden set are never visible to the agent runtime (METR measured a 43× difference in cheating rates when scoring functions are visible — [metr.org, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)).
3. **Iteration circuit breaker**: self-training is capped at 2–3 rounds (Self-Rewarding saturates at round 4, ReST-EM regresses at round 2 — see [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) / [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)); exceeding the limit raises `IterationHalted`.
4. **The golden set never enters training data**: leakage checks (`assert_no_leakage`) run automatically before flywheel export and before every training launch.

---

## Features

EvoFlow's complete capability set, **214 tests green**:

| Capability | Module | Research / Rationale |
|---|---|---|
| Shared LLM client: zero-dependency OpenAI-compatible client + protocol + test stub | `evoflow.llm` (`OpenAICompatClient` / `LLMClient` / `ScriptedLLM`) | Engineering foundation: works with any compatible endpoint (Bailian / DeepSeek / vLLM, etc.); pure-stdlib urllib |
| ACE out-of-the-box loop: LLM reflection + three-role orchestration; refuses to run without a verifier | `l0_context/llm_reflector.py` (`LLMReflector` / `ACELoop`) | ACE, [arXiv:2510.04618](https://arxiv.org/abs/2510.04618) (self-updating degrades without reliable signals) |
| GEPA Pareto prompt evolution with a hard LLM-call budget | `l0_context/gepa.py` (`GEPAEvolver`) | GEPA, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) (authors' self-test: ~10% over GRPO, ~35× fewer rollouts) |
| A-MEM-style episodic memory: automatic link-building, evolving consolidation | `l1_memory/episodic.py` (`EpisodicMemory`) | A-MEM, [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) |
| User-profile memory with `forget` for compliant deletion | `l1_memory/profile.py` (`ProfileMemory`) | Knowledge outside the parameters = deletable, auditable |
| Qdrant production vector backend | `l1_memory/qdrant_store.py` (`QdrantStore`) | Alternative implementation of the `VectorStore` protocol (extra `vec-qdrant`) |
| Knowledge layer: VectorRAG / LightGraphRAG / routing | `evoflow.knowledge` | Contextual Retrieval ([Anthropic, 2024-09](https://www.anthropic.com/news/contextual-retrieval)), LightRAG ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779)), hybrid-retrieval consensus |
| Productionized L2 training backend: config-time red lines + circuit-breaker callbacks | `l2_weight/trainers/qlora_dpo.py` | QLoRA ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)); ⚠️ **main training pipeline not GPU-tested** |
| Echo Trap training-collapse monitor — stops on alarm | `l2_weight/trainers/echo_trap.py` (`EchoTrapMonitor`) | RAGEN, [arXiv:2504.20073](https://arxiv.org/abs/2504.20073) |
| vLLM LoRA hot-swap: rejected with zero HTTP calls if the gate hasn't passed | `l2_weight/serving.py` (`VLLMLoRAServer`) | The engineering closure of "ship the moment the gate passes" |
| Evolution observability: 9 event kinds as a jsonl timeline + composable instrumentation | `evoflow.telemetry` | Engineering building block; OTel / Langfuse export via extra `otel` |
| Judge calibration (Cohen's Kappa) + golden-set leakage guard | `eval/calibration.py` / `eval/leakage.py` | LLM judges agree with humans only to a limited degree — calibrate before going live (Zheng et al., [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)) |
| MCP Server: 4 tools, zero-dependency stdio mode | `evoflow.integrations.mcp_server` | MCP spec (2025-06-18); no golden-set exposure, no promote tool |

> ⚠️ Honest note: GEPA's "~10% over GRPO, 35× fewer rollouts" are the paper authors' self-reported numbers; `LightGraphRAG` is a teaching-grade simplified implementation (marked in its docstring); **the L2 training pipeline has not been GPU-tested end-to-end** — config/data validation and circuit-breaker callbacks are unit-tested, so validate at small scale before ramping up.

---

## Installation

The core package has **zero third-party dependencies** (pure standard library); training / vector / LLM / observability capabilities are optional extras:

```bash
# Minimal install: L0 playbook + L1 built-in JsonlVectorStore (char-ngram hashing vectors, no embedding model needed)
pip install evoflow

# Everything
pip install "evoflow[train,vec,vec-qdrant,llm,otel,mcp]"
```

| Extra | Dependencies | Purpose |
|---|---|---|
| (none) | zero third-party deps | Full closed loop for L0/L1/Evolution Gate/knowledge layer/MCP stdio mode — works out of the box |
| `train` | torch / trl / peft / datasets | L2 training backend (QLoRA + DPO) |
| `vec` | numpy | Vector math support |
| `vec-qdrant` | qdrant-client | `QdrantStore` production vector backend |
| `llm` | openai | Official OpenAI SDK; note: the core `OpenAICompatClient` is pure stdlib and can talk to compatible endpoints without this extra |
| `otel` | opentelemetry-sdk / langfuse | Export the evolution event stream to observability platforms |
| `mcp` | mcp | MCP Server official SDK (FastMCP) enhanced mode |
| `dev` | pytest | Run the test suite |

Requires Python ≥ 3.10.

---

## 5-Minute Quickstart

The examples below use only the core package (zero dependencies) and run the full loop: task → feedback → reflection → playbook evolution → Evolution Gate.

### 1. L0: Context Evolution (ACE-style playbook)

```python
from evoflow.core.trajectory import Trajectory, Outcome
from evoflow.l0_context.playbook import Playbook
from evoflow.l0_context.reflector import Reflector

# 1) Define a reflect function: trajectory in, delta entries out
#    (swap in your own LLM call in production)
def my_reflect(traj: Trajectory):
    if not traj.outcome.success:
        return [{"title": "Confirm the tax rate before quoting", "content": "Always ask whether the price is tax-inclusive",
                 "tags": ["pricing"], "helpful": 1}]
    return []

reflector = Reflector(reflect_fn=my_reflect)
playbook = Playbook()

# 2) One failed task → reflect → curate into the playbook
traj = Trajectory(task="Quote a price for a client", steps=["Quoted a tax-exclusive price directly"],
                  outcome=Outcome(success=False, verifier="user", detail="Client complained"))
playbook.curate(reflector.reflect(traj))

# 3) Inject into the prompt for the next task
system_prompt = playbook.render(max_tokens=800)

# 4) Persist
playbook.save("playbook.json")
```

### 2. L1: Experience Bank (distills successful + failed trajectories, k=1 retrieval)

```python
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore

# JsonlVectorStore: built-in char-ngram hashing vectors, zero dependencies, works out of the box
bank = ReasoningBank(store=JsonlVectorStore("memory.jsonl"), embed_fn=None)

bank.distill(traj)  # both successful and failed trajectories are distilled into StrategyItems

# Retrieve the single most relevant strategy before a new task and inject it into the prompt
# (k=1 is the optimal setting per the ReasoningBank paper)
strategies = bank.retrieve("handle a client's pricing objection", k=1)
```

### 3. Evolution Gate: No Promotion Below the Bar

```python
from evoflow.core.gate import EvolutionGate
from evoflow.eval.golden_set import GoldenSet
from evoflow.eval.judge import RuleBasedJudge

golden = GoldenSet.from_jsonl("golden.jsonl")   # fields: input / expected
gate = EvolutionGate(golden_set=golden, judge=RuleBasedJudge(), min_pass_rate=0.9)

# candidate: any (str) -> str candidate system, e.g. "base prompt + the new playbook"
result = gate.evaluate(candidate=my_agent_fn)
print(result.passed, result.pass_rate)   # discard this update if False
```

### 4. L2: Data Flywheel → QLoRA + DPO (optional, requires `evoflow[train]`)

```python
from evoflow.core.signals import SignalCollector
from evoflow.l2_weight.flywheel import DataFlywheel
from evoflow.l2_weight.registry import AdapterRegistry

flywheel = DataFlywheel(collector=SignalCollector())
pairs = flywheel.build_pairs(signals, trajectories, pos_neg_ratio=2.0)
flywheel.export_dpo_jsonl("dpo_data.jsonl")

# Training (pip install evoflow[train]); the config carries built-in circuit-breaker limits
from evoflow.l2_weight.trainers.qlora_dpo import QLoRADPOConfig, train
config = QLoRADPOConfig(model_name="Qwen/Qwen3-8B", rank=16, epochs=2,
                        lr=1e-5, replay_ratio=0.1, max_rounds=3)
train(config, "dpo_data.jsonl")

# Versioned adapters: promote only if the gate passes, roll back in one command otherwise
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
evoflow init        # scaffold a project (golden.jsonl, playbook.json, directory layout)
evoflow distill     # distill strategies from trajectory logs / build DPO preference pairs
evoflow gate        # evaluate a candidate system against the frozen golden set, output a GateResult
evoflow promote     # promote a given adapter version to production
evoflow rollback    # roll back to the last version that passed the Evolution Gate
```

---

## Documentation

| Doc | Contents |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | Version history (Keep a Changelog format) |
| [docs/architecture.md](docs/architecture.md) | Detailed design of the three-layer evolution stack, data-flow diagrams, design red lines, and research evidence |
| [docs/research-to-code.md](docs/research-to-code.md) | Research-finding → code-module mapping (which class / hard constraint each design decision lands in) |
| [docs/integration.md](docs/integration.md) | LangGraph / CrewAI / Dify integration; connecting Qdrant and OpenAI-compatible APIs |
| [docs/open-source-roadmap.md](docs/open-source-roadmap.md) | Open-source release, CI, versioning strategy, and community roadmap |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributing guide |

## Roadmap

**Current status**: feature-complete — the three-layer evolution stack, Evolution Gate, knowledge layer, training backend, observability, and MCP integration are all in place, with 214 tests green and zero third-party dependencies in the core package. Honest note: the L2 training pipeline has not been GPU-tested end-to-end; config/data validation and circuit-breaker callbacks are unit-tested — please validate on your own hardware at small scale first.

**Directions toward 1.0**:

- **Ecosystem adapters**: LangGraph / CrewAI / Dify adapters, more RAG / GraphRAG integration examples
- **GPU testing**: end-to-end GPU validation of the L2 training pipeline and multi-backend evaluation
- **Documentation**: complete docs and API stabilization, then the 1.0 release

See [docs/open-source-roadmap.md](docs/open-source-roadmap.md) for details.

## Citations & Acknowledgments

EvoFlow's design builds directly on the following public research:

- ACE: Agentic Context Engineering — [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- ReasoningBank — [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
- GEPA: Reflective Prompt Evolution — [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- A-MEM: Agentic Memory — [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- QLoRA — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- RAGEN / StarPO (Echo Trap) — [arXiv:2504.20073](https://arxiv.org/abs/2504.20073)
- Self-Rewarding Language Models — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
- LightRAG — [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
- METR's reward-hacking field study — [metr.org, 2025-06](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
- Self-evolving agents survey — [arXiv:2507.21046](https://arxiv.org/abs/2507.21046)

If you use EvoFlow in research or in a product, please cite this repository (see [CITATION.cff](CITATION.cff)).

## License

[Apache-2.0](LICENSE)
