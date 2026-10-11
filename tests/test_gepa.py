"""GEPA 提示词进化器测试：ScriptedLLM 全程零依赖。

覆盖：Pareto 非支配排序正确性、进化预算耗尽停止、种群截断、
最均衡者返回、迭代熔断纪律接入。
"""

import pytest

from evoflow.l0_context.gepa import GEPAEvolver, PromptCandidate, pareto_front
from evoflow.llm.client import ScriptedLLM


def _c(text: str, **scores) -> PromptCandidate:
    return PromptCandidate(text, dict(scores))


# ---------- pareto_front ----------


def test_pareto_front_drops_dominated():
    a = _c("a", x=0.9, y=0.9)
    b = _c("b", x=0.5, y=0.5)  # 被 a、c 支配
    c = _c("c", x=0.8, y=1.0)  # 与 a 互不支配（y 更优、x 更劣）
    front = pareto_front([a, b, c])
    assert {p.text for p in front} == {"a", "c"}


def test_pareto_front_keeps_tradeoffs():
    p = _c("p", x=1.0, y=0.0)
    q = _c("q", x=0.0, y=1.0)  # 与 p 互不支配
    front = pareto_front([p, q])
    assert {c.text for c in front} == {"p", "q"}


def test_pareto_front_single_dominator():
    best = _c("best", x=1.0, y=1.0, z=1.0)
    others = [_c(f"o{i}", x=0.5, y=0.5, z=0.5) for i in range(3)]
    front = pareto_front([best, *others])
    assert [p.text for p in front] == ["best"]


def test_pareto_front_ties_all_kept():
    a = _c("a", x=0.5)
    b = _c("b", x=0.5)  # 不支配：需至少一个维度严格更优
    assert len(pareto_front([a, b])) == 2


def test_pareto_front_empty():
    assert pareto_front([]) == []


# ---------- GEPAEvolver ----------


def _make_eval(table: dict[str, dict[str, float]]):
    def eval_fn(prompt: str) -> dict[str, float]:
        return dict(table.get(prompt, {"acc": 0.1, "fmt": 0.1}))

    return eval_fn


def test_evolve_budget_exhausted_stops():
    # budget=2：前沿只有种子时第一轮消耗 1 次、第二轮再消耗 1 次后即停
    client = ScriptedLLM(default="variant")
    evolver = GEPAEvolver(client, population_size=8, budget=2)
    eval_fn = _make_eval({"seed": {"acc": 0.5, "fmt": 0.5}, "variant": {"acc": 0.6, "fmt": 0.6}})
    best = evolver.evolve("seed", eval_fn, rounds=100)
    assert evolver.llm_calls <= 2
    assert isinstance(best, PromptCandidate)


def test_evolve_zero_budget_returns_seed():
    client = ScriptedLLM(default="variant")
    evolver = GEPAEvolver(client, budget=0)
    eval_fn = _make_eval({"seed": {"acc": 0.7, "fmt": 0.7}})
    best = evolver.evolve("seed", eval_fn, rounds=5)
    assert best.text == "seed"
    assert evolver.llm_calls == 0


def test_evolve_population_truncated():
    # 每轮从前沿（种子被支配前只有 1 个）产生 1 个变体，容量 3 必须截断
    answers = [f"v{i}" for i in range(10)]
    client = ScriptedLLM(answers=answers, default="vx")
    evolver = GEPAEvolver(client, population_size=3, budget=10)
    table = {"seed": {"acc": 0.2, "fmt": 0.2}}
    for i in range(10):
        table[f"v{i}"] = {"acc": 0.3 + i * 0.01, "fmt": 0.3 + i * 0.01}
    evolver.evolve("seed", _make_eval(table), rounds=10)
    assert len(evolver.population) <= 3


def test_evolve_returns_most_balanced_of_front():
    # 两个互不支配的前沿候选：偏科者 vs 均衡者 → 返回均衡者
    client = ScriptedLLM(answers=["lopsided", "balanced"])
    evolver = GEPAEvolver(client, population_size=8, budget=2)
    eval_fn = _make_eval(
        {
            "seed": {"acc": 0.5, "fmt": 0.5},
            "lopsided": {"acc": 1.0, "fmt": 0.1},  # 与 balanced 互不支配
            "balanced": {"acc": 0.6, "fmt": 0.6},
        }
    )
    best = evolver.evolve("seed", eval_fn, rounds=2)
    assert best.text == "balanced"


def test_evolve_mutation_prompt_names_weakest_dimension():
    client = ScriptedLLM(answers=["v1"])
    evolver = GEPAEvolver(client, budget=1)
    eval_fn = _make_eval({"seed": {"acc": 0.9, "fmt": 0.2}})  # fmt 最弱
    evolver.evolve("seed", eval_fn, rounds=1)
    prompt = client.calls[0][-1]["content"]
    assert "fmt" in prompt and "失分" in prompt


def test_evolve_uses_iteration_controller():
    client = ScriptedLLM(default="v")
    evolver = GEPAEvolver(client, budget=10)
    eval_fn = _make_eval({"seed": {"acc": 0.5}, "v": {"acc": 0.6}})
    evolver.evolve("seed", eval_fn, rounds=3)
    assert evolver.controller is not None
    assert evolver.controller.rounds == 3


def test_evolve_rejects_non_callable_eval_fn():
    evolver = GEPAEvolver(ScriptedLLM(), budget=1)
    with pytest.raises(ValueError):
        evolver.evolve("seed", eval_fn="not-callable", rounds=1)
