"""L2 权重层 —— Adapter 版本化注册表（晋升 / 回滚）。

L2 层设计：每次 QLoRA+DPO 训练产出一个 adapter 版本，注册表统一
登记其指标与评估门结果；**红线：评估门不过线（GateResult.passed=False）
一律禁止晋升**，已晋升版本劣化时可一键回滚到上一版本。

目录结构::

    <root>/
      registry.json     # 版本台账：versions / history / current
      adapters/...      # 各版本 adapter 权重（由训练器写入）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from evoflow.core.gate import GateResult


class PromotionDenied(RuntimeError):
    """晋升被拒绝：评估门未通过（红线硬约束）。"""


class AdapterRegistry:
    """adapter 版本注册表。

    :param root: 注册表根目录。
    """

    INDEX_FILE = "registry.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / self.INDEX_FILE
        self._index = self._load_index()

    # ---------- 持久化 ----------

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with self._index_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"versions": {}, "history": [], "current": None}

    def _save_index(self) -> None:
        with self._index_path.open("w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    # ---------- 版本操作 ----------

    def register(
        self,
        adapter_path: str | Path,
        metrics: dict,
        gate_result: GateResult | None = None,
    ) -> str:
        """登记一个新 adapter 版本（只登记，不晋升）。

        :param adapter_path: adapter 权重路径。
        :param metrics: 训练/评估指标。
        :param gate_result: 可选评估门结果，登记后供 promote 复核。
        :return: 分配的版本号（v0001, v0002, ...）。
        """
        n = len(self._index["versions"]) + 1
        version = f"v{n:04d}"
        self._index["versions"][version] = {
            "adapter_path": str(adapter_path),
            "metrics": metrics,
            "registered_at": time.time(),
            "gate": gate_result.to_dict() if gate_result is not None else None,
        }
        self._save_index()
        return version

    def promote(self, version: str, gate_result: GateResult | None = None) -> str:
        """晋升指定版本为当前版本。

        红线：必须提供（或登记时留存）评估门结果且 ``passed=True``，
        否则抛 :class:`PromotionDenied`。

        :param version: 版本号。
        :param gate_result: 评估门结果；缺省用登记时留存的结果。
        """
        record = self._index["versions"].get(version)
        if record is None:
            raise KeyError(f"未知版本: {version}")
        gr = gate_result
        if gr is None and record.get("gate") is not None:
            g = record["gate"]
            gr = GateResult(
                passed=g["passed"],
                pass_rate=g["pass_rate"],
                failures=g.get("failures", []),
            )
        if gr is None or not gr.passed:
            raise PromotionDenied(
                f"评估门未通过（version={version}），红线约束：禁止晋升"
            )
        current = self._index["current"]
        if current is not None and current != version:
            self._index["history"].append(current)
        self._index["current"] = version
        self._save_index()
        return version

    def rollback(self) -> str | None:
        """回滚到上一版本。

        :return: 恢复后的版本号；无历史版本时返回 None（当前版本保持不变）。
        """
        if not self._index["history"]:
            return None
        previous = self._index["history"].pop()
        self._index["current"] = previous
        self._save_index()
        return previous

    # ---------- 查询 ----------

    @property
    def current(self) -> str | None:
        """当前版本号（未晋升过则为 None）。"""
        return self._index["current"]

    def current_path(self) -> Path | None:
        """当前版本 adapter 路径。"""
        cur = self.current
        if cur is None:
            return None
        return Path(self._index["versions"][cur]["adapter_path"])

    def versions(self) -> list[str]:
        """全部已登记版本号。"""
        return list(self._index["versions"].keys())

    def info(self, version: str) -> dict:
        """查询版本详情。"""
        return self._index["versions"][version]
