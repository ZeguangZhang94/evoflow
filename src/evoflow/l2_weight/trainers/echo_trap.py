"""L2 权重层 —— Echo Trap 监控器（多轮 agent RL 塌缩检测）。

对应 RAGEN 论文（arXiv:2504.20073）观察到的「Echo Trap」：多轮 agent RL
训练中，奖励方差塌缩 -> 熵降 -> 梯度尖峰 -> 策略同质化（所有轨迹趋同，
模型只会"复读"同一策略），训练继续只会浪费算力并放大漂移。

本模块实现三个早期信号检测，任一触发即报警；:meth:`EchoTrapMonitor.enforce`
在报警时抛 :class:`EchoTrapHalted`，供训练循环调用，对应设计红线「掉即停」。

纯标准库实现，可完整测试。
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field


@dataclass
class EchoTrapStatus:
    """Echo Trap 检测结果。

    :param ok: 是否健康（无任何信号触发）。
    :param signals: 触发的信号描述列表（ok 时为空）。
    """

    ok: bool
    signals: list[str] = field(default_factory=list)


class EchoTrapHalted(RuntimeError):
    """Echo Trap 熔断异常：检测到塌缩信号即抛出（红线：掉即停）。"""


class EchoTrapMonitor:
    """Echo Trap 滚动窗口监控器。

    三个信号（任一触发即报警）：

    1. **奖励方差塌缩**：窗口内各 rollout 批次的奖励方差均值 < ``var_floor``
       （所有轨迹奖励趋同，优势信号消失）；
    2. **熵塌缩**：最新熵低于 ``entropy_floor`` 且在窗口内持续下降
       （策略确定性化，探索消失）；
    3. **梯度尖峰**：最新 grad_norm 超过窗口历史均值的 ``grad_spike`` 倍
       （策略剧烈跳变，常伴随塌缩前兆）。

    :param window: 滚动窗口大小（记录的批次数）。
    :param var_floor: 奖励方差地板（默认 1e-3）。
    :param entropy_floor: 熵地板（默认 0.1）。
    :param grad_spike: 梯度尖峰倍数（默认 5.0，即超过历史均值 5 倍）。
    """

    def __init__(
        self,
        window: int = 8,
        var_floor: float = 1e-3,
        entropy_floor: float = 0.1,
        grad_spike: float = 5.0,
    ) -> None:
        if window < 3:
            raise ValueError("window 必须 >= 3（趋势判断至少需要 3 个点）")
        if var_floor < 0:
            raise ValueError("var_floor 必须 >= 0")
        if entropy_floor < 0:
            raise ValueError("entropy_floor 必须 >= 0")
        if grad_spike <= 1.0:
            raise ValueError("grad_spike 必须 > 1（否则正常波动也会误报）")
        self.window = window
        self.var_floor = var_floor
        self.entropy_floor = entropy_floor
        self.grad_spike = grad_spike
        self._reward_vars: deque[float] = deque(maxlen=window)
        self._entropies: deque[float] = deque(maxlen=window)
        self._grad_norms: deque[float] = deque(maxlen=window)

    @property
    def recorded(self) -> int:
        """已记录的批次数（达到 window 后保持 window）。"""
        return len(self._grad_norms)

    def record(self, rewards: list[float], entropy: float, grad_norm: float) -> None:
        """记录一批 rollout 的统计量。

        :param rewards: 本批各条轨迹的奖励（至少 2 条，否则无法计算方差）。
        :param entropy: 本批策略熵。
        :param grad_norm: 本步梯度范数。
        """
        if len(rewards) < 2:
            raise ValueError("rewards 至少需要 2 个样本才能计算奖励方差")
        self._reward_vars.append(statistics.pvariance([float(r) for r in rewards]))
        self._entropies.append(float(entropy))
        self._grad_norms.append(float(grad_norm))

    def check(self) -> EchoTrapStatus:
        """检查三个塌缩信号，任一触发即报警。"""
        signals: list[str] = []

        # 信号 1：奖励方差塌缩（窗口满才判断，避免冷启动误报）
        if len(self._reward_vars) == self.window:
            mean_var = statistics.fmean(self._reward_vars)
            if mean_var < self.var_floor:
                signals.append(
                    f"奖励方差塌缩：窗口均值 {mean_var:.3e} < "
                    f"var_floor={self.var_floor:.3e}（轨迹奖励趋同，优势信号消失）"
                )

        # 信号 2：熵低于地板且窗口内持续下降
        entropies = list(self._entropies)
        if (
            len(entropies) >= 3
            and entropies[-1] < self.entropy_floor
            and all(b < a for a, b in zip(entropies, entropies[1:], strict=False))
        ):
            signals.append(
                f"熵塌缩：熵持续下降 {entropies[0]:.4f} -> {entropies[-1]:.4f} "
                f"且低于 entropy_floor={self.entropy_floor}（策略探索消失）"
            )

        # 信号 3：梯度尖峰（最新值超过历史均值的 grad_spike 倍）
        grads = list(self._grad_norms)
        if len(grads) >= 3:
            baseline = statistics.fmean(grads[:-1])
            if baseline > 0 and grads[-1] > self.grad_spike * baseline:
                signals.append(
                    f"梯度尖峰：grad_norm={grads[-1]:.4f} 超过历史均值 "
                    f"{baseline:.4f} 的 {self.grad_spike} 倍（策略剧烈跳变）"
                )

        return EchoTrapStatus(ok=not signals, signals=signals)

    def enforce(self) -> EchoTrapStatus:
        """红线「掉即停」：任一信号触发即抛 :class:`EchoTrapHalted`。

        供训练循环在每步/每轮调用；健康时返回 :class:`EchoTrapStatus`。
        """
        status = self.check()
        if not status.ok:
            raise EchoTrapHalted(
                "Echo Trap 红线触发（掉即停，训练必须中止并人工介入）："
                + "；".join(status.signals)
            )
        return status
