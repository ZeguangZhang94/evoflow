"""L0 上下文进化 —— LLM 驱动的 Reflector 与 ACE 编排循环。

对应 ACE（Agentic Context Engineering，arXiv:2510.04618）的
Generator -> Reflector -> Curator 三角色循环的开箱即用实现：

- :class:`LLMReflector`：用 LLM 分析轨迹（成功 → 提炼可复用做法；
  失败 → 提炼教训），以固定行格式 ``- [标签] 内容`` 输出并解析为
  :class:`DeltaEntry`。解析失败时回退为整条文本一条 delta；
  LLM 调用异常时返回空列表 —— **绝不抛异常中断进化循环**。
- :class:`ACELoop`：编排器。渲染 playbook → 跑业务 agent → verifier
  判定 → reflect → curate 入库。对应 ACE 论文"无 GT 标签时靠执行反馈
  （execution feedback）驱动上下文进化"的机制，verifier 即执行反馈。

红线（硬约束）：没有可靠 verifier 时 :meth:`ACELoop.run` **拒绝执行**
（raise ValueError）。ACE 论文（arXiv:2510.04618）指出，上下文自我更新
依赖可靠反馈信号；无可靠反馈时自我更新会退化（自我强化的错误被不断
固化进上下文），因此宁可不进化也不盲目进化。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable

from evoflow.core.trajectory import Outcome, Trajectory
from evoflow.l0_context.playbook import DeltaEntry, Playbook
from evoflow.llm.client import LLMClient

# 固定输出格式：每行一条 "- [标签] 内容"
_LINE_RE = re.compile(r"^\s*[-*•]\s*\[([^\[\]]+)\]\s*(\S.*)$")

_SYSTEM_PROMPT = (
    "你是 ACE（Agentic Context Engineering）框架中的 Reflector。"
    "你的任务是分析一条 agent 任务轨迹，提炼可增量累积进 playbook 的条目。"
    "严格要求：每条洞察独占一行，格式为 `- [标签] 内容`，"
    "标签用简短名词（如 策略、陷阱、格式、工具），不要输出任何其他文字。"
)


class LLMReflector:
    """LLM 驱动的反思者（ACE 的 Reflector 角色，arXiv:2510.04618）。

    :param client: 满足 :class:`LLMClient` 协议的模型调用方。
    :param max_deltas: 单次反思最多产出的 delta 条数（默认 3，
        防止一次反思灌入过多噪声条目）。
    """

    def __init__(self, client: LLMClient, max_deltas: int = 3) -> None:
        if max_deltas < 1:
            raise ValueError("max_deltas 必须 >= 1")
        self.client = client
        self.max_deltas = max_deltas

    def reflect(self, trajectory: Trajectory) -> list[DeltaEntry]:
        """对一条轨迹反思，返回 DeltaEntry 列表（未入库，交由 Playbook.curate）。

        成功轨迹提炼可复用做法，失败轨迹提炼教训；解析失败回退为整条文本
        一条 delta；LLM 调用异常返回空列表。本方法绝不抛异常。
        """
        try:
            raw = self.client.chat(self._build_messages(trajectory))
        except Exception:
            # 反思是进化循环的一环，LLM 抖动不应中断整个循环
            return []
        return self._parse(raw, success=trajectory.outcome.success)

    def _build_messages(self, trajectory: Trajectory) -> list[dict]:
        outcome = trajectory.outcome
        if outcome.success:
            goal = "该轨迹【成功】。请提炼可复用的做法（什么策略导致了成功）。"
        else:
            goal = "该轨迹【失败】。请提炼教训（什么导致了失败、下次如何避免）。"
        steps = "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(trajectory.steps)
        ) or "（无步骤记录）"
        user = (
            f"## 任务\n{trajectory.task}\n\n"
            f"## 轨迹步骤\n{steps}\n\n"
            f"## 结局\nverifier={outcome.verifier}，detail={outcome.detail or '无'}\n\n"
            f"{goal}\n"
            f"最多输出 {self.max_deltas} 条，每条一行，格式 `- [标签] 内容`。"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _parse(self, raw: str, success: bool) -> list[DeltaEntry]:
        """解析固定格式输出；失败时回退为整条文本一条 delta。"""
        outcome_tag = "success" if success else "fail"
        deltas: list[DeltaEntry] = []
        for line in (raw or "").splitlines():
            m = _LINE_RE.match(line)
            if not m:
                continue
            labels = [t.strip() for t in re.split(r"[,，/]", m.group(1)) if t.strip()]
            content = m.group(2).strip()
            if not content:
                continue
            deltas.append(
                DeltaEntry(
                    id=f"d-{uuid.uuid4().hex[:12]}",
                    content=content,
                    tags=labels + ["reflect", outcome_tag],
                )
            )
            if len(deltas) >= self.max_deltas:
                return deltas
        if deltas:
            return deltas
        # 回退：模型未按格式输出时，整条文本作为一条 delta，不丢失信号
        text = (raw or "").strip()
        if not text:
            return []
        return [
            DeltaEntry(
                id=f"d-{uuid.uuid4().hex[:12]}",
                content=text,
                tags=["reflect", outcome_tag, "unparsed"],
            )
        ]


class ACELoop:
    """ACE 三角色编排器：Generator -> Reflector -> Curator（arXiv:2510.04618）。

    对应 ACE 论文"无 GT 标签时靠执行反馈驱动进化"的机制：业务 agent 充当
    Generator（消费 playbook 文本完成任务），``verifier`` 充当执行反馈
    （execution feedback，如单元测试、规则校验、环境奖励），Reflector 从
    被验证的轨迹中提炼 delta，Curator（:meth:`Playbook.curate`）入库。

    :param agent_fn: 业务 agent callable，入参 ``(task, playbook_text)``，
        返回最终回答文本。
    :param reflector: 任何具有 ``reflect(trajectory) -> list[DeltaEntry]``
        方法的反思者（如 :class:`LLMReflector` 或规则版 ``Reflector``）。
    :param playbook: 增量 playbook（Curator 角色内建于其 ``curate`` 方法）。
    """

    def __init__(
        self,
        agent_fn: Callable[[str, str], str],
        reflector,
        playbook: Playbook,
    ) -> None:
        if not callable(agent_fn):
            raise ValueError("agent_fn 必须是 callable")
        if not hasattr(reflector, "reflect"):
            raise ValueError("reflector 必须提供 reflect(trajectory) 方法")
        self.agent_fn = agent_fn
        self.reflector = reflector
        self.playbook = playbook

    def run(
        self,
        task: str,
        verifier: Callable[[str], bool] | None = None,
    ) -> Trajectory:
        """执行一轮完整 ACE 循环，返回被验证的轨迹（副作用：playbook 已入库）。

        :param task: 任务描述。
        :param verifier: 执行反馈 callable，输入最终回答、返回是否成功。
        :raises ValueError: verifier 缺失或不可调用。ACE 论文
            （arXiv:2510.04618）表明上下文自我更新依赖可靠反馈信号；
            无可靠反馈时自我更新会退化（错误被自我强化固化进上下文），
            因此本方法宁可拒绝执行也不做无反馈的盲目进化。
        """
        if verifier is None or not callable(verifier):
            raise ValueError(
                "ACELoop 拒绝在无可靠 verifier 时运行：ACE（arXiv:2510.04618）"
                "指出自我更新依赖可靠执行反馈，无反馈的自我更新会退化。"
                "请提供 verifier（如单元测试、规则校验或环境奖励）作为执行反馈。"
            )
        playbook_text = self.playbook.render()
        answer = self.agent_fn(task, playbook_text)
        success = bool(verifier(answer))
        trajectory = Trajectory(
            task=task,
            steps=[answer],
            outcome=Outcome(
                success=success,
                verifier=getattr(verifier, "__name__", "callable"),
                detail="verifier 判定通过" if success else "verifier 判定未通过",
            ),
        )
        deltas = self.reflector.reflect(trajectory)
        self.playbook.curate(deltas)
        return trajectory
