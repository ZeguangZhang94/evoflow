"""L2 权重层 —— QLoRA + DPO 训练器（生产级完整实现）。

L2 层设计（详见 docs/research-to-code.md）：用数据飞轮产出的偏好对做 DPO 训练，QLoRA（4bit 量化 +
LoRA 低秩适配）控制显存；``replay_ratio=0.1`` 混入 10% 旧样本防止灾难性
遗忘；训练轮数受 :class:`evoflow.core.discipline.IterationController`
熔断约束（超轮或指标连续回退即抛 ``IterationHalted``）。

红线约束（在 :class:`QLoRADPOConfig.__post_init__` 硬校验）：

- ``epochs <= 3``：设计红线「epoch≤3」，超过即 ``ValueError``，不接受警告放行；
- ``rank <= 32``：LoRA 秩上限，超过即 ``ValueError``；
- 其余超参（lr/batch_size/grad_accum/max_seq_len/replay_ratio/max_rounds）
  做区间校验，非法即拒绝。

依赖红线：torch / trl / peft / datasets / transformers 为 extras 重依赖，
**运行时惰性导入**；缺失时抛 :class:`TrainDependencyError`（LLMError 风格），
提示 ``请 pip install evoflow[train]``。核心包零第三方依赖承诺不受影响。

注意：训练主流程（:func:`train` 主体）需要 GPU 与训练依赖，本仓库测试只覆盖
零依赖部分（config 校验 / 数据加载 / 回放混合 / 缺依赖报错）；带
``@pytest.mark.train`` 的用例在安装 ``evoflow[train]`` 的环境才会运行。
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from evoflow.core.discipline import IterationController
from evoflow.llm.client import LLMError

_INSTALL_HINT = "缺少训练依赖：请 pip install evoflow[train]"

#: DPO jsonl 每行必须具备的字段（与 DataFlywheel.export_dpo_jsonl 对齐）
REQUIRED_PAIR_KEYS = ("prompt", "chosen", "rejected")

#: LoRA 注入的线性层（LLaMA/Qwen 等主流 decoder-only 架构通用命名）
_DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

#: 设计红线：epoch 上限
MAX_EPOCHS = 3
#: 设计红线：LoRA 秩上限
MAX_RANK = 32


class TrainDependencyError(LLMError):
    """训练重依赖缺失（torch/trl/peft/datasets/transformers）。

    继承 :class:`evoflow.llm.client.LLMError`，保持项目统一的 LLMError 风格；
    报错信息固定包含 ``请 pip install evoflow[train]`` 的安装指引。
    """


@dataclass
class QLoRADPOConfig:
    """QLoRA + DPO 训练配置（构造即校验，红线不过直接拒绝）。

    :param model_name: 基座模型名或本地路径。
    :param rank: LoRA 秩（红线：<= 32）。
    :param lora_alpha: LoRA 缩放系数（惯例为 rank 的 2 倍）。
    :param epochs: 每轮训练 epoch 数（红线：<= 3）。
    :param lr: 学习率。
    :param batch_size: 单设备批大小。
    :param grad_accum: 梯度累积步数（有效批量 = batch_size * grad_accum）。
    :param max_seq_len: 最大序列长度（prompt + completion）。
    :param replay_ratio: 旧样本回放比例（防遗忘，默认 0.1 即 10%）。
    :param max_rounds: 自训练最大轮数（熔断上限，建议 2~3）。
    :param output_dir: adapter 输出目录。
    :param seed: 随机种子。
    """

    model_name: str
    rank: int = 16
    lora_alpha: int = 32
    epochs: int = 2
    lr: float = 1e-5
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1024
    replay_ratio: float = 0.1
    max_rounds: int = 3
    output_dir: str = "adapters"
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model_name or not str(self.model_name).strip():
            raise ValueError("model_name 不能为空")
        if self.epochs > MAX_EPOCHS:
            raise ValueError(
                f"epochs={self.epochs} 超过红线 MAX_EPOCHS={MAX_EPOCHS}"
                "（设计红线：epoch≤3，防止过拟合与漂移），拒绝训练配置"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs={self.epochs} 必须 >= 1")
        if self.rank > MAX_RANK:
            raise ValueError(
                f"rank={self.rank} 超过红线 MAX_RANK={MAX_RANK}"
                "（LoRA 秩上限，防止低秩假设失效与显存失控），拒绝训练配置"
            )
        if self.rank < 1:
            raise ValueError(f"rank={self.rank} 必须 >= 1")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha={self.lora_alpha} 必须 >= 1")
        if self.lr <= 0:
            raise ValueError(f"lr={self.lr} 必须 > 0")
        if self.batch_size < 1:
            raise ValueError(f"batch_size={self.batch_size} 必须 >= 1")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum={self.grad_accum} 必须 >= 1")
        if self.max_seq_len < 16:
            raise ValueError(f"max_seq_len={self.max_seq_len} 必须 >= 16")
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError(
                f"replay_ratio={self.replay_ratio} 必须在 [0, 1] 区间内"
            )
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds={self.max_rounds} 必须 >= 1")


def _lazy_imports() -> dict:
    """惰性导入训练重依赖，缺失时抛出清晰安装指引。

    :return: {"torch", "datasets", "peft", "transformers", "trl"} 模块字典。
    :raises TrainDependencyError: 任一依赖缺失，提示 ``pip install evoflow[train]``。
    """
    modules = {}
    for name in ("torch", "datasets", "peft", "transformers", "trl"):
        try:
            modules[name] = __import__(name)
        except ImportError as e:
            raise TrainDependencyError(
                f"{_INSTALL_HINT}（缺失模块: {name}）"
            ) from e
    return modules


def load_dpo_pairs(path: str | Path) -> list[dict]:
    """加载飞轮导出的 DPO jsonl（prompt / chosen / rejected）。

    :param path: jsonl 路径（空行自动跳过）。
    :return: 偏好对字典列表。
    :raises ValueError: 某行缺少必需字段（报出行号，便于修复飞轮导出）。
    """
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_PAIR_KEYS if not row.get(k)]
            if missing:
                raise ValueError(
                    f"{path}:{lineno} 缺少必需字段 {missing}，"
                    f"要求字段: {REQUIRED_PAIR_KEYS}"
                )
            rows.append(
                {
                    "prompt": str(row["prompt"]),
                    "chosen": str(row["chosen"]),
                    "rejected": str(row["rejected"]),
                }
            )
    return rows


# 向后兼容别名（旧骨架导出名）
load_pairs = load_dpo_pairs


def mix_replay(
    new_pairs: list[dict],
    old_pairs: list[dict],
    replay_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """混入旧样本防灾难性遗忘：回放量 = replay_ratio * 新样本数（纯标准库）。

    :param new_pairs: 本轮新偏好对。
    :param old_pairs: 历史偏好对池。
    :param replay_ratio: 回放比例（默认 0.1，即 10%）。
    :param seed: 抽样种子（固定种子保证可复现）。
    :return: 新样本 + 抽样旧样本，整体打乱；旧池为空或比例 <= 0 时原样返回新样本。
    """
    if not old_pairs or replay_ratio <= 0:
        return list(new_pairs)
    n_replay = min(len(old_pairs), max(1, int(len(new_pairs) * replay_ratio)))
    rng = random.Random(seed)
    mixed = list(new_pairs) + rng.sample(old_pairs, n_replay)
    rng.shuffle(mixed)
    return mixed


def train(
    config: QLoRADPOConfig,
    data_path: str | Path,
    replay_path: str | Path | None = None,
    on_round_end: Callable[[float], None] | None = None,
) -> str:
    """执行一轮 QLoRA + DPO 训练（需要 GPU 与 ``evoflow[train]`` 依赖）。

    流程：加载偏好对 -> 混入回放样本 -> 4bit BitsAndBytes 量化加载基座 ->
    PEFT LoraConfig 注入 -> TRL DPOTrainer 训练 -> 每 epoch 结束记录熔断指标
    并回调 ``on_round_end`` -> 保存 adapter。

    :param config: 训练配置（构造时已通过红线校验）。
    :param data_path: 飞轮导出的 DPO jsonl 路径。
    :param replay_path: 可选的历史数据 jsonl，用于回放防遗忘。
    :param on_round_end: 每 epoch 结束回调，签名为 ``(metric: float) -> None``，
        metric 越大越好（取 loss 的相反数）。
    :return: adapter 输出路径（``<output_dir>/last``）。
    :raises TrainDependencyError: 缺少训练依赖（提示 ``pip install evoflow[train]``）。
    :raises evoflow.core.discipline.IterationHalted: 触发迭代熔断（超轮/连续回退）。
    """
    mods = _lazy_imports()
    torch = mods["torch"]
    # 以下代码仅在安装 evoflow[train] 后执行；核心包测试不覆盖本函数主体。
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import DPOConfig, DPOTrainer

    new_pairs = load_dpo_pairs(data_path)
    old_pairs = load_dpo_pairs(replay_path) if replay_path else []
    if not new_pairs:
        raise ValueError(f"{data_path} 中没有可用的偏好对，拒绝空训练")
    mixed = mix_replay(new_pairs, old_pairs, config.replay_ratio, seed=config.seed)

    # 迭代熔断纪律：每 epoch 记一轮指标，超 max_rounds 或连续回退即抛 IterationHalted。
    controller = IterationController(max_rounds=config.max_rounds)

    # ---- 4bit QLoRA 量化加载基座 ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # 训练态关闭 KV cache，配合梯度检查点省显存
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(_DEFAULT_TARGET_MODULES),
    )

    dataset = Dataset.from_list(mixed)

    class _RoundEndCallback(TrainerCallback):
        """每 epoch 结束：记录熔断指标并回调 on_round_end（metric 越大越好）。"""

        def on_epoch_end(self, args, state, control, **kwargs):
            metric = 0.0
            for log in reversed(state.log_history):
                if "eval_loss" in log:
                    metric = -float(log["eval_loss"])
                    break
                if "loss" in log:
                    metric = -float(log["loss"])
                    break
            # IterationController 超轮/连续回退时抛 IterationHalted，中止训练。
            controller.record_round(metric)
            if on_round_end is not None:
                on_round_end(metric)
            return control

    ckpt_dir = Path(config.output_dir)
    args = DPOConfig(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=0.1,
        max_length=config.max_seq_len,
        max_prompt_length=max(8, config.max_seq_len // 2),
        seed=config.seed,
        logging_steps=10,
        save_strategy="no",  # checkpoint 由本函数显式保存，避免中间态落盘
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    )

    trainer_kwargs = dict(
        model=model,  # ref_model=None：PEFT 模式下 TRL 自动用禁用 adapter 的基座做参考模型
        args=args,
        train_dataset=dataset,
        peft_config=lora,
        callbacks=[_RoundEndCallback()],
    )
    try:  # trl >= 0.12
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:  # trl 0.8 ~ 0.11 旧签名
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()

    adapter_dir = ckpt_dir / "last"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))  # PEFT 模型只保存 adapter 权重
    tokenizer.save_pretrained(str(adapter_dir))

    return str(adapter_dir)
