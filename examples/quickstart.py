"""EvoFlow 零依赖端到端 quickstart。

模拟一个客服 agent 的一天，串起三层进化栈的完整闭环：

1. 反馈采集（贯穿层）   —— SignalCollector 落盘 accept/reject/correction
2. L1 经验蒸馏          —— ReasoningBank 从成功+失败轨迹蒸馏策略并检索（k=1）
3. L0 上下文进化        —— Reflector 反思 -> Playbook 增量 curate -> render
4. L2 数据飞轮          —— DataFlywheel 产出 DPO 偏好对并导出
5. 评估门裁决（红线）   —— 冻结黄金集回归：好候选 PROMOTE，人为构造一轮
                           失败展示 REJECT，带基线时展示 ROLLBACK
6. 迭代熔断（红线）     —— IterationController 超轮抛 IterationHalted

运行：``python examples/quickstart.py``（无需安装任何第三方依赖）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 未安装包时直接从 src 布局导入，保证裸环境可跑。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoflow import (
    AdapterRegistry,
    DataFlywheel,
    EvolutionGate,
    FeedbackSignal,
    GateResult,
    GoldenSet,
    IterationController,
    IterationHalted,
    JsonlVectorStore,
    Outcome,
    Playbook,
    ReasoningBank,
    Reflector,
    RuleBasedJudge,
    SignalCollector,
    SignalType,
    Trajectory,
)

OUT = Path(__file__).resolve().parent / "_demo_out"


def hr(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------------------------------------------------------- 模拟客服 agent
FAQ = {
    "如何重置密码？": "您可以在「设置-账号安全」中重置密码，按短信验证码确认即可。",
    "如何申请退款？": "请在订单详情页点击「申请退款」，客服将在 24 小时内处理您的退款。",
}


def customer_service_agent(question: str) -> str:
    """模拟客服 agent：命中 FAQ 则回答，否则给出敷衍回答（会失败）。"""
    return FAQ.get(question, "不太清楚，建议您再看看帮助文档。")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- 0. 模拟一天的任务流
    hr("0. 模拟客服 agent 任务流")
    tasks = ["如何重置密码？", "如何申请退款？", "你们支持以旧换新吗？"]
    trajectories: list[Trajectory] = []
    collector = SignalCollector(OUT / "signals.jsonl")
    for task in tasks:
        answer = customer_service_agent(task)
        success = task in FAQ
        outcome = Outcome(
            success=success,
            verifier="user",
            detail="用户满意" if success else "用户不满：回答敷衍、未解决问题",
        )
        traj = Trajectory(task=task, steps=[f"检索 FAQ: {task}", answer], outcome=outcome)
        trajectories.append(traj)
        collector.emit(
            SignalType.SUCCESS if success else SignalType.FAIL,
            task_id=task,
            value=1.0 if success else -1.0,
        )
        print(f"  任务: {task} -> {'成功' if success else '失败'} | {answer[:30]}...")

    # 用户亲手纠正了一个坏回答（飞轮中最高质量的信号）
    collector.collect(
        FeedbackSignal(
            type=SignalType.CORRECTION,
            task_id="你们支持以旧换新吗？",
            value=1.0,
            meta={
                "task": "你们支持以旧换新吗？",
                "original": "不太清楚，建议您再看看帮助文档。",
                "corrected": (
                    "支持以旧换新：请在「服务-以旧换新」提交旧机信息，估价后可抵扣新机款。"
                ),
            },
        )
    )
    print(f"  信号采集完毕：{len(collector)} 条（含 1 条用户纠正）")

    # ---------------------------------------------------------- 1. L1 ReasoningBank
    hr("1. L1 经验库：成功+失败轨迹都蒸馏")
    bank = ReasoningBank(JsonlVectorStore())
    for traj in trajectories:
        item = bank.distill(traj)
        print(f"  蒸馏[{item.meta['kind']}]: {item.title}")
    hits = bank.retrieve("旧手机以旧换新怎么办理", k=1)
    print(f"  检索(k=1, 相关性>数量): {hits[0].title}")
    print(f"    内容摘要: {hits[0].content.splitlines()[-1]}")

    # ---------------------------------------------------------- 2. L0 Playbook 进化
    hr("2. L0 上下文进化：Reflector -> Curator -> Playbook")

    def simple_reflect(traj: Trajectory) -> list[str]:
        if traj.outcome.success:
            return [f"命中 FAQ 的问题要直接给操作路径（如：{traj.task}）"]
        return [
            f"对未覆盖的问题（如：{traj.task}）禁止敷衍回答，应承认盲区并给出人工入口",
            "回答必须包含具体入口路径，不要只说「看帮助文档」",
        ]

    reflector = Reflector(simple_reflect)
    playbook = Playbook()
    for traj in trajectories:
        playbook.curate(reflector.reflect(traj))
    # 再 curate 一次相同内容，验证去重合并（条目数不变）
    added = playbook.curate(reflector.reflect(trajectories[-1]))
    print(f"  去重后 playbook 共 {len(playbook)} 条（重复 curate 新增 {len(added)} 条）")
    print("  render 进 system prompt 的效果：")
    for line in playbook.render(max_tokens=200).splitlines():
        print(f"    {line}")
    playbook.save(OUT / "playbook.json")

    # ---------------------------------------------------------- 3. L2 数据飞轮
    hr("3. L2 数据飞轮：信号+轨迹 -> DPO 偏好对")
    flywheel = DataFlywheel(collector)
    pairs = flywheel.build_pairs(trajectories=trajectories, pos_neg_ratio=2.0)
    for p in pairs:
        print(f"  任务: {p.task}")
        print(f"    chosen  : {p.chosen[:40]}")
        print(f"    rejected: {p.rejected[:40]}")
    n = flywheel.export_dpo_jsonl(OUT / "dpo_train.jsonl")
    print(f"  已导出 {n} 条偏好对 -> {OUT / 'dpo_train.jsonl'}（纠正对优先、已去重）")

    # ---------------------------------------------------------- 4. 评估门裁决（红线）
    hr("4. 评估门：冻结黄金集回归测试，不过线不晋升")
    golden_path = OUT / "golden.jsonl"
    with golden_path.open("w", encoding="utf-8") as f:
        f.write('{"input": "如何重置密码？", "expected": "重置密码"}\n')
        f.write('{"input": "如何申请退款？", "expected": "退款"}\n')
    golden = GoldenSet.from_jsonl(golden_path)
    judge = RuleBasedJudge(mode="contains")

    # 候选 A：当前 agent（能过线）
    gate = EvolutionGate(golden, judge, min_pass_rate=0.9)
    good = gate.evaluate(customer_service_agent)
    print(f"  候选A（当前 agent）: pass_rate={good.pass_rate:.0%} -> {good.decision.value}")

    # 候选 B：人为构造的失败候选（无基线 -> REJECT）
    def broken_agent(question: str) -> str:
        return "系统维护中，请稍后再试。"

    bad = gate.evaluate(broken_agent)
    print(
        f"  候选B（人为失败）: pass_rate={bad.pass_rate:.0%} "
        f"-> {bad.decision.value}（不过线禁止晋升）"
    )

    # 候选 B 再评一次，这次带基线 -> ROLLBACK
    gate_with_base = EvolutionGate(
        golden, judge, min_pass_rate=0.9, baseline=customer_service_agent
    )
    rb = gate_with_base.evaluate(broken_agent)
    print(
        f"  候选B（带基线）  : pass_rate={rb.pass_rate:.0%} "
        f"-> {rb.decision.value}（回滚到已晋升版本）"
    )

    # 注册表红线演示：坏候选的 GateResult 无法晋升
    registry = AdapterRegistry(OUT / "adapters")
    version = registry.register(OUT / "adapters" / "v0001", metrics={"pass_rate": 0.0})
    try:
        registry.promote(version, bad)
    except Exception as e:  # PromotionDenied
        print(f"  注册表红线生效: {type(e).__name__}: {e}")
    registry.promote(version, GateResult(passed=True, pass_rate=1.0, decision=good.decision))
    print(f"  好结果可晋升: current={registry.current}")

    # ---------------------------------------------------------- 5. 迭代熔断（红线）
    hr("5. 迭代熔断：自训练最多 2~3 轮")
    controller = IterationController(max_rounds=3)
    try:
        for metric in [0.80, 0.85, 0.90, 0.92]:  # 第 4 轮必然熔断
            controller.record_round(metric)
            print(f"  第 {controller.rounds} 轮: metric={metric:.2f}")
    except IterationHalted as e:
        print(f"  熔断触发: {e}")

    hr("quickstart 完成")
    print(f"产物目录: {OUT}")


if __name__ == "__main__":
    main()
