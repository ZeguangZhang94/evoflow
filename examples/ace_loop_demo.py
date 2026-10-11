"""EvoFlow ACELoop（ACE 三角色循环）零依赖演示。

演示内容：

- Generator：一个假的 ``agent_fn``（消费 playbook 文本、产出回答）；
- Reflector：``LLMReflector`` + ``ScriptedLLM``（预置反思答案，无需联网）；
- Curator：``Playbook.curate`` 增量入库；
- verifier：规则判分（回答必须包含指定关键词），充当 ACE 论文所说的
  "执行反馈"（execution feedback）；
- 跑 3 个任务，观察 playbook 条目数随任务增长（增量 delta 进化）；
- 红线演示：缺 verifier 时 ``ACELoop.run`` 直接拒绝执行。

运行：``python examples/ace_loop_demo.py``（零依赖）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 未安装包时直接从 src 布局导入，保证裸环境可跑。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoflow import ACELoop, LLMReflector, Playbook, ScriptedLLM
from evoflow.core.trajectory import Trajectory


def hr(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ------------------------------------------------------------------ 假 agent
ANSWERS = {
    "回复客户退款进度咨询": "您好，您的退款已验收通过，退款将在 3 个工作日内原路退回。",
    "起草新品上架公告": "新品上架公告：本周五全新系列上架，首批限量发售，敬请关注。",
    "解释 API 429 报错": "HTTP 429 表示触发限流：默认 60 次/分钟，请指数退避重试。",
}


def fake_agent(task: str, playbook_text: str) -> str:
    """假 Generator：命中题库则回答，并在回答头部注入 playbook 行数，
    便于观察"上下文进化被消费"这件事。"""
    answer = ANSWERS.get(task, "抱歉，这个问题我暂时无法回答。")
    return f"[playbook:{len(playbook_text.splitlines())}行] {answer}"


# ------------------------------------------------------------------ 规则 verifier
def make_keyword_verifier(keyword: str):
    """规则 verifier：回答包含关键词即成功（执行反馈的零依赖替身）。"""

    def verifier(answer: str) -> bool:
        return keyword in answer

    verifier.__name__ = f"keyword_verifier({keyword})"
    return verifier


def main() -> None:
    # ScriptedLLM 预置 3 次反思输出（固定格式：每行 `- [标签] 内容`）。
    reflect_llm = ScriptedLLM(
        answers=[
            "- [策略] 退款类问题先给到账时限，再给查询入口\n- [格式] 客户回复控制在三段以内",
            "- [策略] 公告类文案必须包含时间点与行动号召",
            "- [陷阱] 解释报错时先给结论再展开原因，避免用户读到一半流失",
        ],
        default="- [策略] 无新经验",
    )
    reflector = LLMReflector(reflect_llm)
    playbook = Playbook()
    loop = ACELoop(agent_fn=fake_agent, reflector=reflector, playbook=playbook)

    tasks = [
        ("回复客户退款进度咨询", "退款"),
        ("起草新品上架公告", "上架"),
        ("解释 API 429 报错", "429"),
    ]

    hr("ACELoop：3 个任务，playbook 随任务进化")
    for i, (task, keyword) in enumerate(tasks, start=1):
        before = len(playbook)
        traj: Trajectory = loop.run(task, verifier=make_keyword_verifier(keyword))
        added = len(playbook) - before
        print(f"\n  第 {i} 轮 任务: {task}")
        print(f"    agent 回答: {traj.steps[0]}")
        print(f"    verifier={traj.outcome.verifier} -> success={traj.outcome.success}")
        print(f"    playbook: {before} 条 -> {len(playbook)} 条（新增 {added} 条 delta）")
        for entry in playbook.entries[before:]:
            print(f"      + [{', '.join(entry.tags)}] {entry.content}")

    hr("进化后的 playbook（render 进 system prompt 的文本）")
    for line in playbook.render(max_tokens=500).splitlines():
        print(f"    {line}")

    hr("红线：无 verifier 拒绝执行")
    try:
        loop.run("任意任务")  # 缺 verifier -> ValueError
    except ValueError as e:
        print(f"  ACELoop 拒绝运行: {type(e).__name__}: {str(e)[:60]}...")

    hr("ace loop demo 完成")


if __name__ == "__main__":
    main()
