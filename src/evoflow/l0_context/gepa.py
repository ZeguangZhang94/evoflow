"""L0 上下文进化 —— GEPA 提示词进化器（反思式 Pareto 种群进化）。

对应 GEPA（Genetic-Pareto，斯坦福 + MIT 2025，arXiv:2507.19457）：
用**自然语言反思**替代标量奖励做突变 —— 对 Pareto 前沿上的提示词候选，
让 LLM 针对其失分维度生成改写变体；种群经多目标 Pareto 非支配排序
截断维护。论文结果：同等预算下超过 GRPO 平均约 10%，且 rollout 用量
省约 35 倍 —— 因此本实现把 LLM 调用次数作为一等公民的 ``budget``
硬约束，耗尽即停。

红线（评分器隔离）：:class:`GEPAEvolver` 只接受 ``eval_fn`` callable
（``prompt -> {维度: 分数}``），不访问、也不允许访问任何 judge 内部
实现；评分细节除了用于生成变异提示（"哪个维度失分"）外不作他用。

纯标准库实现，:func:`pareto_front` 可独立测试。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from evoflow.core.discipline import IterationController, IterationHalted
from evoflow.llm.client import LLMClient


@dataclass
class PromptCandidate:
    """提示词候选个体。

    :param text: 提示词文本。
    :param scores: 多目标评分，``{评测维度: 分数}``，所有维度均为越大越好。
    """

    text: str
    scores: dict[str, float] = field(default_factory=dict)


def _dims(candidates: Sequence[PromptCandidate]) -> list[str]:
    """评分维度的并集（稳定顺序，缺失维度按 0.0 计）。"""
    seen: list[str] = []
    for c in candidates:
        for k in c.scores:
            if k not in seen:
                seen.append(k)
    return seen


def _dominates(a: PromptCandidate, b: PromptCandidate, dims: Sequence[str]) -> bool:
    """a 支配 b：所有维度不劣且至少一个维度严格更优。"""
    av = [a.scores.get(d, 0.0) for d in dims]
    bv = [b.scores.get(d, 0.0) for d in dims]
    return all(x >= y for x, y in zip(av, bv, strict=True)) and any(
        x > y for x, y in zip(av, bv, strict=True)
    )


def pareto_front(candidates: Sequence[PromptCandidate]) -> list[PromptCandidate]:
    """标准 Pareto 非支配排序的第一前沿（所有互不支配的候选）。

    所有评分维度均视为越大越好；缺失维度按 0.0 计。
    """
    fronts = _non_dominated_sort(candidates)
    return fronts[0] if fronts else []


def _non_dominated_sort(
    candidates: Sequence[PromptCandidate],
) -> list[list[PromptCandidate]]:
    """快速非支配排序：返回按层级排列的全部前沿。"""
    dims = _dims(candidates)
    remaining = list(candidates)
    fronts: list[list[PromptCandidate]] = []
    while remaining:
        front = [
            c
            for c in remaining
            if not any(o is not c and _dominates(o, c, dims) for o in remaining)
        ]
        fronts.append(front)
        remaining = [c for c in remaining if c not in front]
    return fronts


def _mean_score(c: PromptCandidate) -> float:
    return sum(c.scores.values()) / len(c.scores) if c.scores else 0.0


def _most_balanced(front: Sequence[PromptCandidate]) -> PromptCandidate:
    """Pareto 前沿中的最均衡者：最大化最差维度（max-min），平局取均值高者。"""
    dims = _dims(front)

    def key(c: PromptCandidate) -> tuple[float, float]:
        values = [c.scores.get(d, 0.0) for d in dims]
        return (min(values) if values else 0.0, _mean_score(c))

    return max(front, key=key)


class GEPAEvolver:
    """GEPA 提示词进化器：反思变异 + Pareto 前沿种群（arXiv:2507.19457）。

    :param client: 满足 :class:`LLMClient` 协议的模型调用方（仅用于反思变异）。
    :param population_size: 种群容量，每轮结束后按非支配排序截断到该值。
    :param budget: 总 LLM 调用预算（只计反思变异调用；对应 GEPA"rollout
        省 35 倍"的成本意识），耗尽即停止进化并返回当前最优。
    """

    def __init__(
        self,
        client: LLMClient,
        population_size: int = 8,
        budget: int = 100,
    ) -> None:
        if population_size < 1:
            raise ValueError("population_size 必须 >= 1")
        if budget < 0:
            raise ValueError("budget 必须 >= 0")
        self.client = client
        self.population_size = population_size
        self.budget = budget
        self._spent = 0
        self._population: list[PromptCandidate] = []
        self.controller: IterationController | None = None

    @property
    def llm_calls(self) -> int:
        """已消耗的 LLM 调用次数。"""
        return self._spent

    @property
    def population(self) -> list[PromptCandidate]:
        """当前种群（只读副本）。"""
        return list(self._population)

    def evolve(
        self,
        seed_prompt: str,
        eval_fn: Callable[[str], dict[str, float]],
        rounds: int,
    ) -> PromptCandidate:
        """从种子提示词进化 ``rounds`` 轮，返回当前 Pareto 前沿中的最均衡者。

        每轮：对 Pareto 前沿候选逐个做 LLM 反思变异（告知其在哪个维度
        失分，请改写）→ ``eval_fn`` 评估变体 → 合并种群 → 非支配排序
        截断到 ``population_size``。预算耗尽或迭代熔断即提前停止。

        :param seed_prompt: 种子提示词。
        :param eval_fn: 评分器 callable，``prompt -> {维度: 分数}``。
            评分器隔离红线：本方法只调用它，绝不探查其内部实现。
        :param rounds: 计划进化轮数（经 :class:`IterationController` 熔断控制）。
        """
        if not callable(eval_fn):
            raise ValueError("eval_fn 必须是 callable（评分器隔离：只接受 callable）")
        if rounds < 0:
            raise ValueError("rounds 必须 >= 0")
        self.controller = IterationController(max_rounds=max(1, rounds))
        self._population = [PromptCandidate(seed_prompt, dict(eval_fn(seed_prompt)))]
        for _ in range(rounds):
            if self._spent >= self.budget:
                break
            offspring = self._mutate_front(eval_fn)
            if offspring:
                self._population.extend(offspring)
                self._population = self._truncate(self._population)
            metric = max(_mean_score(c) for c in self._population)
            try:
                self.controller.record_round(metric)
            except IterationHalted:
                # 指标连续回退：进化方向失效，按迭代熔断纪律停止
                break
        return _most_balanced(pareto_front(self._population))

    def _mutate_front(
        self, eval_fn: Callable[[str], dict[str, float]]
    ) -> list[PromptCandidate]:
        """对当前 Pareto 前沿逐个反思变异（受预算约束）。"""
        offspring: list[PromptCandidate] = []
        for cand in pareto_front(self._population):
            if self._spent >= self.budget:
                break
            variant = self._reflect_mutate(cand)
            if variant and variant.strip() and variant.strip() != cand.text.strip():
                offspring.append(
                    PromptCandidate(variant.strip(), dict(eval_fn(variant.strip())))
                )
        return offspring

    def _reflect_mutate(self, cand: PromptCandidate) -> str:
        """GEPA 式自然语言反思变异：针对最弱维度请 LLM 改写提示词。"""
        self._spent += 1
        if cand.scores:
            weakest = min(cand.scores, key=lambda d: cand.scores[d])
            weakest_line = (
                f"这段提示词在「{weakest}」维度失分最多"
                f"（得分 {cand.scores[weakest]}），各维度得分：{cand.scores}。"
            )
        else:
            weakest_line = "这段提示词尚无有效评分。"
        user = (
            "你是提示词进化中的反思变异算子（GEPA）。\n"
            f"{weakest_line}\n"
            "请改写下面的提示词，重点提升失分维度，同时不损害其他维度。"
            "只输出改写后的提示词全文，不要输出解释。\n\n"
            f"## 原提示词\n{cand.text}"
        )
        return self.client.chat([{"role": "user", "content": user}])

    def _truncate(
        self, population: Sequence[PromptCandidate]
    ) -> list[PromptCandidate]:
        """非支配排序后按层级截断到 population_size（层级内保均值高者）。"""
        kept: list[PromptCandidate] = []
        for front in _non_dominated_sort(population):
            room = self.population_size - len(kept)
            if room <= 0:
                break
            kept.extend(sorted(front, key=_mean_score, reverse=True)[:room])
        return kept
