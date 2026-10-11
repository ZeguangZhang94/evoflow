"""QLoRA+DPO 训练器测试：零依赖部分全覆盖；训练主流程标记 @pytest.mark.train。"""

import importlib.util
import json
import sys

import pytest

from evoflow.l2_weight.trainers.qlora_dpo import (
    QLoRADPOConfig,
    TrainDependencyError,
    load_dpo_pairs,
    mix_replay,
    train,
)
from evoflow.llm.client import LLMError

TRAIN_DEPS = ("torch", "datasets", "peft", "transformers", "trl")
TRAIN_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in TRAIN_DEPS)
needs_train = pytest.mark.skipif(not TRAIN_AVAILABLE, reason="未安装 evoflow[train] 依赖")


def _pair(i: int, suffix: str = "") -> dict:
    return {
        "prompt": f"问题{i}",
        "chosen": f"好答案{i}{suffix}",
        "rejected": f"坏答案{i}{suffix}",
    }


def _write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


# ---------- QLoRADPOConfig 红线校验 ----------


class TestConfig:
    def test_defaults_are_valid(self):
        cfg = QLoRADPOConfig(model_name="Qwen/Qwen2.5-7B-Instruct")
        assert cfg.rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.epochs == 2
        assert cfg.lr == 1e-5
        assert cfg.batch_size == 4
        assert cfg.grad_accum == 4
        assert cfg.max_seq_len == 1024
        assert cfg.replay_ratio == 0.1
        assert cfg.max_rounds == 3
        assert cfg.seed == 42

    def test_epochs_red_line(self):
        """红线：epochs > 3 一律拒绝；边界 3 放行。"""
        QLoRADPOConfig(model_name="m", epochs=3)  # 边界放行
        with pytest.raises(ValueError, match="epochs"):
            QLoRADPOConfig(model_name="m", epochs=4)

    def test_rank_red_line(self):
        """红线：rank > 32 一律拒绝；边界 32 放行。"""
        QLoRADPOConfig(model_name="m", rank=32)  # 边界放行
        with pytest.raises(ValueError, match="rank"):
            QLoRADPOConfig(model_name="m", rank=64)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_name": ""},
            {"model_name": "m", "epochs": 0},
            {"model_name": "m", "rank": 0},
            {"model_name": "m", "lr": 0},
            {"model_name": "m", "batch_size": 0},
            {"model_name": "m", "grad_accum": 0},
            {"model_name": "m", "max_seq_len": 8},
            {"model_name": "m", "replay_ratio": -0.1},
            {"model_name": "m", "replay_ratio": 1.1},
            {"model_name": "m", "max_rounds": 0},
        ],
    )
    def test_invalid_params_rejected(self, kwargs):
        with pytest.raises(ValueError):
            QLoRADPOConfig(**kwargs)


# ---------- load_dpo_pairs ----------


class TestLoadDpoPairs:
    def test_parses_flywheel_jsonl(self, tmp_path):
        path = tmp_path / "dpo.jsonl"
        rows = [_pair(1), _pair(2)]
        _write_jsonl(path, rows)
        # 追加空行，应被跳过
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n   \n")
        pairs = load_dpo_pairs(path)
        assert pairs == rows

    def test_missing_field_raises_with_lineno(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        _write_jsonl(path, [_pair(1), {"prompt": "q", "chosen": "c"}])  # 缺 rejected
        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            load_dpo_pairs(path)

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_dpo_pairs(path) == []


# ---------- mix_replay ----------


class TestMixReplay:
    def test_replay_ratio(self):
        new = [_pair(i) for i in range(100)]
        old = [_pair(i, suffix="_old") for i in range(50)]
        mixed = mix_replay(new, old, replay_ratio=0.1)
        assert len(mixed) == 110  # 100 新 + 10 旧
        for p in new:
            assert p in mixed  # 新样本全保留
        replayed = [p for p in mixed if p not in new]
        assert len(replayed) == 10
        for p in replayed:
            assert p in old  # 回放样本全部来自旧池

    def test_zero_ratio_or_empty_pool_returns_new_only(self):
        new = [_pair(i) for i in range(10)]
        old = [_pair(i, suffix="_old") for i in range(5)]
        assert mix_replay(new, old, replay_ratio=0.0) == new
        assert mix_replay(new, [], replay_ratio=0.1) == new

    def test_replay_capped_by_pool_size(self):
        new = [_pair(i) for i in range(10)]
        old = [_pair(i, suffix="_old") for i in range(2)]
        mixed = mix_replay(new, old, replay_ratio=0.9)
        assert len(mixed) == 10 + 2  # 需求 9 条回放，被旧池大小 2 封顶

    def test_deterministic_with_seed(self):
        new = [_pair(i) for i in range(20)]
        old = [_pair(i, suffix="_old") for i in range(20)]
        assert mix_replay(new, old, 0.2, seed=7) == mix_replay(new, old, 0.2, seed=7)


# ---------- train() 缺依赖报错（monkeypatch 模拟 import 失败） ----------


class TestTrainDependencyGuard:
    def test_missing_torch_raises_clear_error(self, monkeypatch, tmp_path):
        data = tmp_path / "dpo.jsonl"
        _write_jsonl(data, [_pair(1)])
        # sys.modules 置 None：即使环境装了 torch，import 也会抛 ImportError
        monkeypatch.setitem(sys.modules, "torch", None)
        cfg = QLoRADPOConfig(model_name="dummy-model")
        with pytest.raises(TrainDependencyError, match=r"pip install evoflow\[train\]"):
            train(cfg, data)

    def test_error_is_llm_error_style(self, monkeypatch, tmp_path):
        data = tmp_path / "dpo.jsonl"
        _write_jsonl(data, [_pair(1)])
        monkeypatch.setitem(sys.modules, "datasets", None)
        cfg = QLoRADPOConfig(model_name="dummy-model")
        with pytest.raises(LLMError):  # TrainDependencyError 继承 LLMError
            train(cfg, data)


# ---------- 需要 evoflow[train] 真实依赖的用例（本机 skip） ----------


@needs_train
@pytest.mark.train
def test_lazy_imports_resolve_when_deps_present():
    """安装 evoflow[train] 的环境下，惰性导入应成功解析五个重依赖。"""
    from evoflow.l2_weight.trainers import qlora_dpo

    mods = qlora_dpo._lazy_imports()
    for name in ("torch", "datasets", "peft", "transformers", "trl"):
        assert mods[name] is not None
