"""迭代熔断模块（贯穿层：迭代熔断纪律）。

红线硬约束：自训练/自进化迭代必须在 max_rounds（建议 2~3 轮）内停止，
指标连续回退同样触发熔断。超过即抛 :class:`IterationHalted`，防止无限
自训练导致的漂移与算力浪费。
"""

from __future__ import annotations


class IterationHalted(RuntimeError):
    """迭代熔断异常：超轮或指标连续回退时抛出。"""


class IterationController:
    """迭代控制器。

    :param max_rounds: 最大允许轮数（默认 3，建议 2~3）。
    :param max_consecutive_regressions: 允许的最大连续回退次数（默认 2）。
        连续回退达到该值即熔断。
    """

    def __init__(self, max_rounds: int = 3, max_consecutive_regressions: int = 2) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须 >= 1")
        if max_consecutive_regressions < 1:
            raise ValueError("max_consecutive_regressions 必须 >= 1")
        self.max_rounds = max_rounds
        self.max_consecutive_regressions = max_consecutive_regressions
        self.history: list[float] = []

    @property
    def rounds(self) -> int:
        """已记录轮数。"""
        return len(self.history)

    def record_round(self, metric: float) -> None:
        """记录一轮指标并检查熔断条件。

        :param metric: 本轮评估指标（越大越好，如黄金集通过率）。
        :raises IterationHalted: 超过 max_rounds，或指标连续回退达到上限。
        """
        self.history.append(float(metric))
        if len(self.history) > self.max_rounds:
            raise IterationHalted(
                f"迭代轮数 {len(self.history)} 超过 max_rounds={self.max_rounds}，"
                "触发熔断：自进化必须停止"
            )
        regressions = self._trailing_regressions()
        if regressions >= self.max_consecutive_regressions:
            raise IterationHalted(
                f"指标连续回退 {regressions} 次（历史: {self.history}），"
                "触发熔断：进化方向失效，停止并人工介入"
            )

    def _trailing_regressions(self) -> int:
        """统计末尾连续回退次数。"""
        n = 0
        for i in range(len(self.history) - 1, 0, -1):
            if self.history[i] < self.history[i - 1]:
                n += 1
            else:
                break
        return n
