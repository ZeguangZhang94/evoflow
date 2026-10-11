"""LLMReflector 与 ACELoop 测试：ScriptedLLM 全程零依赖。

覆盖：reflect 解析成功 / 解析失败回退 / max_deltas 截断 / LLM 异常不中断、
ACELoop 完整闭环（成功与失败轨迹）、无 verifier 拒绝执行。
"""

import pytest

from evoflow.core.trajectory import Outcome, Trajectory
from evoflow.l0_context.llm_reflector import ACELoop, LLMReflector
from evoflow.l0_context.playbook import Playbook
from evoflow.llm.client import LLMError, ScriptedLLM


def _traj(success: bool) -> Trajectory:
    return Trajectory(
        task="总结这篇论文",
        steps=["先读摘要", "论文提出三层进化栈"],
        outcome=Outcome(success, "rule", "detail"),
    )


# ---------- LLMReflector ----------


def test_reflect_parses_structured_lines():
    client = ScriptedLLM(answers=["- [策略] 先读摘要再总结\n- [格式] 输出要分点"])
    ref = LLMReflector(client)
    deltas = ref.reflect(_traj(success=True))
    assert len(deltas) == 2
    assert deltas[0].content == "先读摘要再总结"
    assert "策略" in deltas[0].tags
    assert "reflect" in deltas[0].tags and "success" in deltas[0].tags
    assert deltas[1].content == "输出要分点"


def test_reflect_fail_trajectory_gets_fail_tag():
    client = ScriptedLLM(answers=["- [陷阱] 不要编造引用"])
    ref = LLMReflector(client)
    deltas = ref.reflect(_traj(success=False))
    assert len(deltas) == 1
    assert "fail" in deltas[0].tags


def test_reflect_fallback_when_unparseable():
    prose = "这次失败主要是因为没有核对来源，下次应该先列大纲。"
    client = ScriptedLLM(answers=[prose])
    ref = LLMReflector(client)
    deltas = ref.reflect(_traj(success=False))
    assert len(deltas) == 1
    assert deltas[0].content == prose
    assert "unparsed" in deltas[0].tags


def test_reflect_respects_max_deltas():
    client = ScriptedLLM(answers=["- [a] 一\n- [b] 二\n- [c] 三\n- [d] 四"])
    ref = LLMReflector(client, max_deltas=2)
    deltas = ref.reflect(_traj(success=True))
    assert len(deltas) == 2
    assert [d.content for d in deltas] == ["一", "二"]


def test_reflect_llm_error_returns_empty_and_never_raises():
    class BoomLLM:
        def chat(self, messages, **kwargs):
            raise LLMError("网络抖动")

    ref = LLMReflector(BoomLLM())
    assert ref.reflect(_traj(success=True)) == []


def test_reflect_prompt_mentions_success_or_fail_goal():
    client = ScriptedLLM(answers=["- [策略] x"])
    LLMReflector(client).reflect(_traj(success=True))
    prompt = client.calls[0][-1]["content"]
    assert "成功" in prompt and "可复用" in prompt


# ---------- ACELoop ----------


def _agent(task: str, playbook_text: str) -> str:
    return f"回答：{task}"


def test_ace_loop_full_cycle_success():
    client = ScriptedLLM(answers=["- [策略] 先复述任务再作答"])
    playbook = Playbook()
    loop = ACELoop(agent_fn=_agent, reflector=LLMReflector(client), playbook=playbook)
    traj = loop.run("总结论文", verifier=lambda ans: ans.startswith("回答"))
    assert traj.outcome.success is True
    assert traj.final_answer == "回答：总结论文"
    assert len(playbook) == 1  # curate 已入库
    assert playbook.entries[0].content == "先复述任务再作答"


def test_ace_loop_fail_trajectory_also_curated():
    client = ScriptedLLM(answers=["- [陷阱] 不要遗漏约束条件"])
    playbook = Playbook()
    loop = ACELoop(agent_fn=_agent, reflector=LLMReflector(client), playbook=playbook)
    traj = loop.run("总结论文", verifier=lambda ans: False)
    assert traj.outcome.success is False
    assert len(playbook) == 1
    assert "fail" in playbook.entries[0].tags


def test_ace_loop_renders_playbook_into_agent():
    seen: list[str] = []

    def spy_agent(task: str, playbook_text: str) -> str:
        seen.append(playbook_text)
        return "ok"

    playbook = Playbook()
    loop = ACELoop(agent_fn=spy_agent, reflector=LLMReflector(ScriptedLLM()), playbook=playbook)
    loop.run("t1", verifier=lambda ans: True)
    loop.run("t2", verifier=lambda ans: True)
    assert "Playbook" in seen[0]
    assert seen[0] == playbook.render()


def test_ace_loop_rejects_missing_verifier():
    loop = ACELoop(agent_fn=_agent, reflector=LLMReflector(ScriptedLLM()), playbook=Playbook())
    with pytest.raises(ValueError, match="verifier"):
        loop.run("任务", verifier=None)
    with pytest.raises(ValueError):
        loop.run("任务")  # 默认即拒绝
    with pytest.raises(ValueError):
        loop.run("任务", verifier="not-callable")
