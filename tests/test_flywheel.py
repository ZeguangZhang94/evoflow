"""L2 数据飞轮测试：纠正优先、成败配对、去重、比例控制、导出。"""

import json

import pytest

from evoflow.core.signals import FeedbackSignal, SignalCollector, SignalType
from evoflow.core.trajectory import Outcome, Trajectory
from evoflow.l2_weight.flywheel import DataFlywheel


def _traj(task: str, success: bool, answer: str) -> Trajectory:
    return Trajectory(
        task=task,
        steps=["思考", answer],
        outcome=Outcome(success=success, verifier="user"),
    )


def test_correction_signal_has_priority():
    collector = SignalCollector()
    collector.collect(
        FeedbackSignal(
            type=SignalType.CORRECTION,
            task_id="t1",
            value=1.0,
            meta={"task": "怎么退货", "original": "不知道", "corrected": "订单页申请退货"},
        )
    )
    flywheel = DataFlywheel(collector)
    pairs = flywheel.build_pairs(trajectories=[])
    assert len(pairs) == 1
    assert pairs[0].chosen == "订单页申请退货"
    assert pairs[0].rejected == "不知道"
    assert pairs[0].task == "怎么退货"


def test_success_fail_pairing_same_task():
    flywheel = DataFlywheel()
    trajs = [
        _traj("重置密码", True, "设置里重置"),
        _traj("重置密码", False, "不知道"),
    ]
    pairs = flywheel.build_pairs(signals=[], trajectories=trajs)
    assert len(pairs) == 1
    assert pairs[0].chosen == "设置里重置"
    assert pairs[0].rejected == "不知道"


def test_dedup_identical_pairs():
    flywheel = DataFlywheel()
    trajs = [
        _traj("重置密码", True, "设置里重置"),
        _traj("重置密码", True, "设置里重置"),  # 重复成功
        _traj("重置密码", False, "不知道"),
        _traj("重置密码", False, "不知道"),  # 重复失败
    ]
    pairs = flywheel.build_pairs(signals=[], trajectories=trajs)
    assert len(pairs) == 1


def test_pos_neg_ratio_caps_rejected():
    flywheel = DataFlywheel()
    trajs = [_traj("任务", True, "好答案1"), _traj("任务", True, "好答案2")]
    trajs += [_traj("任务", False, f"坏答案{i}") for i in range(10)]
    pairs = flywheel.build_pairs(signals=[], trajectories=trajs, pos_neg_ratio=2.0)
    # 正:负 = 2:1 -> 2 个 chosen 最多配 1 个 rejected（负样本不得超过正样本的一半）
    assert len(pairs) == 1


def test_no_pair_without_both_sides():
    flywheel = DataFlywheel()
    pairs = flywheel.build_pairs(
        signals=[], trajectories=[_traj("任务", True, "只有成功")]
    )
    assert pairs == []


def test_missing_collector_and_signals_raises():
    with pytest.raises(ValueError):
        DataFlywheel().build_pairs(trajectories=[])


def test_export_dpo_jsonl(tmp_path):
    flywheel = DataFlywheel()
    flywheel.build_pairs(
        signals=[],
        trajectories=[_traj("重置密码", True, "设置里重置"), _traj("重置密码", False, "不知道")],
    )
    path = tmp_path / "dpo.jsonl"
    n = flywheel.export_dpo_jsonl(path)
    assert n == 1
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row == {"prompt": "重置密码", "chosen": "设置里重置", "rejected": "不知道"}
