"""L0 上下文进化层（类 ACE）：Generator -> Reflector -> Curator 的 playbook 进化。

开箱即用的 LLM 驱动三角色循环见 :class:`LLMReflector` 与 :class:`ACELoop`
（ACE，arXiv:2510.04618）；提示词种群进化见 :class:`GEPAEvolver`
（GEPA，arXiv:2507.19457）。
"""

from evoflow.l0_context.gepa import GEPAEvolver, PromptCandidate, pareto_front
from evoflow.l0_context.llm_reflector import ACELoop, LLMReflector
from evoflow.l0_context.playbook import DeltaEntry, Playbook
from evoflow.l0_context.reflector import Reflector

__all__ = [
    "ACELoop",
    "DeltaEntry",
    "GEPAEvolver",
    "LLMReflector",
    "Playbook",
    "PromptCandidate",
    "Reflector",
    "pareto_front",
]
