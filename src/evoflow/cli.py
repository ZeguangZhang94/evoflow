"""EvoFlow 命令行入口（贯穿各层的运维工具）。

子命令：
- ``init``     ：生成项目骨架（data/golden.jsonl、adapters/ 等）。
- ``distill``  ：从轨迹 jsonl 蒸馏进 ReasoningBank（L1）。
- ``gate``     ：用冻结黄金集评估候选 callable，输出裁决（评估门）。
- ``promote``  ：晋升 adapter 版本（评估门不过线则拒绝，红线）。
- ``rollback`` ：回滚到上一 adapter 版本。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from evoflow.core.gate import EvolutionGate, GateResult
from evoflow.core.trajectory import Trajectory
from evoflow.eval.golden_set import GoldenSet
from evoflow.eval.judge import RuleBasedJudge
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied

_SAMPLE_GOLDEN = [
    {"input": "示例：如何重置密码？", "expected": "重置密码"},
    {"input": "示例：如何申请退款？", "expected": "退款"},
]


def _cmd_init(args: argparse.Namespace) -> int:
    """生成项目骨架目录。"""
    root = Path(args.dir)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "adapters").mkdir(parents=True, exist_ok=True)
    golden = root / "data" / "golden.jsonl"
    if not golden.exists():
        with golden.open("w", encoding="utf-8") as f:
            for row in _SAMPLE_GOLDEN:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("trajectories.jsonl", "signals.jsonl", "reasoning_bank.jsonl"):
        p = root / "data" / name
        if not p.exists():
            p.touch()
    print(f"已生成 EvoFlow 项目骨架于 {root.resolve()}")
    print("  data/golden.jsonl        冻结黄金集（评估门回归测试）")
    print("  data/trajectories.jsonl  任务轨迹（L0/L1 原料）")
    print("  data/signals.jsonl       反馈信号（L2 飞轮原料）")
    print("  adapters/                adapter 版本注册表根目录")
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    """从轨迹 jsonl 蒸馏进 ReasoningBank。"""
    store = JsonlVectorStore()
    store.load(args.bank)
    bank = ReasoningBank(store)
    n = 0
    with Path(args.trajectories).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = bank.distill(Trajectory.from_dict(json.loads(line)))
            n += 1
            print(f"  蒸馏[{item.meta.get('kind', '?')}]: {item.title}")
    store.save(args.bank)
    print(f"共蒸馏 {n} 条策略，记忆库总量 {len(bank)}，已保存至 {args.bank}")
    return 0


def _load_callable(spec: str):
    """按 'module:function' 规格加载 callable。"""
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"callable 规格必须是 'module:function'，收到: {spec}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _cmd_gate(args: argparse.Namespace) -> int:
    """运行评估门。"""
    golden = GoldenSet.from_jsonl(args.golden)
    judge = RuleBasedJudge(mode=args.judge_mode)
    baseline = _load_callable(args.baseline) if args.baseline else None
    gate = EvolutionGate(golden, judge, min_pass_rate=args.min_pass_rate, baseline=baseline)
    candidate = _load_callable(args.candidate)
    result = gate.evaluate(candidate)
    print(f"通过率: {result.pass_rate:.2%}（阈值 {args.min_pass_rate:.0%}）")
    print(f"裁决: {result.decision.value}")
    for fail in result.failures:
        print(
            f"  未过用例: input={fail['input']!r} "
            f"expected={fail['expected']!r} output={fail['output']!r}"
        )
    return 0 if result.passed else 1


def _gate_result_from_args(args: argparse.Namespace) -> GateResult:
    passed = args.pass_rate >= args.min_pass_rate
    return GateResult(passed=passed, pass_rate=args.pass_rate)


def _cmd_promote(args: argparse.Namespace) -> int:
    """晋升 adapter 版本（红线：评估门不过线禁止晋升）。"""
    registry = AdapterRegistry(args.registry)
    gate_result = _gate_result_from_args(args)
    try:
        registry.promote(args.version, gate_result)
    except PromotionDenied as e:
        print(f"晋升被拒绝: {e}", file=sys.stderr)
        return 1
    print(f"已晋升 {args.version} 为当前版本（pass_rate={args.pass_rate:.2%}）")
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    """回滚到上一 adapter 版本。"""
    registry = AdapterRegistry(args.registry)
    previous = registry.rollback()
    if previous is None:
        print("无历史版本可回滚", file=sys.stderr)
        return 1
    print(f"已回滚到 {previous}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="evoflow", description="EvoFlow —— 三层进化栈的自进化 Agent 框架"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="生成项目骨架目录")
    p_init.add_argument("dir", nargs="?", default=".", help="目标目录（默认当前目录）")
    p_init.set_defaults(func=_cmd_init)

    p_distill = sub.add_parser("distill", help="从轨迹蒸馏进 ReasoningBank（L1）")
    p_distill.add_argument("--trajectories", required=True, help="轨迹 jsonl 路径")
    p_distill.add_argument("--bank", required=True, help="记忆库 jsonl 路径")
    p_distill.set_defaults(func=_cmd_distill)

    p_gate = sub.add_parser("gate", help="评估门：冻结黄金集回归测试")
    p_gate.add_argument("--golden", required=True, help="黄金集 jsonl 路径")
    p_gate.add_argument("--candidate", required=True, help="候选 callable，格式 module:function")
    p_gate.add_argument(
        "--baseline", default=None, help="基线 callable（候选不过线时裁决 ROLLBACK）"
    )
    p_gate.add_argument("--min-pass-rate", type=float, default=0.9, help="过线阈值（默认 0.9）")
    p_gate.add_argument("--judge-mode", choices=["contains", "exact"], default="contains")
    p_gate.set_defaults(func=_cmd_gate)

    p_promote = sub.add_parser("promote", help="晋升 adapter 版本（评估门不过线则拒绝）")
    p_promote.add_argument("--registry", required=True, help="注册表根目录")
    p_promote.add_argument("--version", required=True, help="版本号，如 v0001")
    p_promote.add_argument("--pass-rate", type=float, required=True, help="评估门实测通过率")
    p_promote.add_argument("--min-pass-rate", type=float, default=0.9, help="过线阈值（默认 0.9）")
    p_promote.set_defaults(func=_cmd_promote)

    p_rollback = sub.add_parser("rollback", help="回滚到上一 adapter 版本")
    p_rollback.add_argument("--registry", required=True, help="注册表根目录")
    p_rollback.set_defaults(func=_cmd_rollback)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
