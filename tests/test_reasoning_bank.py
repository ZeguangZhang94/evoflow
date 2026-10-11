"""L1 ReasoningBank 测试：成功/失败都蒸馏、检索 k=1、store 持久化。"""

from evoflow.core.trajectory import Outcome, Trajectory
from evoflow.l1_memory.reasoning_bank import ReasoningBank
from evoflow.l1_memory.store import JsonlVectorStore


def _traj(task: str, success: bool, answer: str) -> Trajectory:
    return Trajectory(
        task=task,
        steps=[f"处理 {task}", answer],
        outcome=Outcome(success=success, verifier="user", detail="详情"),
    )


def test_distill_success_and_failure_both_stored():
    bank = ReasoningBank(JsonlVectorStore())
    ok = bank.distill(_traj("重置密码", True, "去设置里重置"))
    bad = bank.distill(_traj("以旧换新", False, "不知道"))
    assert len(bank) == 2
    assert ok.meta["kind"] == "success"
    assert bad.meta["kind"] == "fail"
    assert "失败原因" in bad.content


def test_retrieve_default_k1_returns_most_relevant():
    bank = ReasoningBank(JsonlVectorStore())
    bank.distill(_traj("如何重置密码", True, "设置-账号安全里重置密码"))
    bank.distill(_traj("如何申请退款", True, "订单页点申请退款"))
    hits = bank.retrieve("密码忘记了怎么重置")  # 默认 k=1
    assert len(hits) == 1
    assert "密码" in hits[0].title


def test_retrieve_k2():
    bank = ReasoningBank(JsonlVectorStore())
    bank.distill(_traj("如何重置密码", True, "设置里重置密码"))
    bank.distill(_traj("如何申请退款", True, "申请退款"))
    assert len(bank.retrieve("如何操作", k=2)) == 2


def test_store_save_load_roundtrip(tmp_path):
    path = tmp_path / "bank.jsonl"
    bank = ReasoningBank(JsonlVectorStore())
    bank.distill(_traj("重置密码", True, "去设置"))
    bank.store.save(path)

    store2 = JsonlVectorStore()
    store2.load(path)
    bank2 = ReasoningBank(store2)
    assert len(bank2) == 1
    assert "密码" in bank2.retrieve("重置密码")[0].title
