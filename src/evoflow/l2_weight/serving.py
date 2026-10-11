"""L2 权重层 —— vLLM LoRA adapter 热切换（评估门过线即上线）。

L2 层「评估门过线即上线」的设计：训练产出的 adapter 版本经
:class:`evoflow.l2_weight.registry.AdapterRegistry` 登记与评估门晋升后，
通过 vLLM 的运行时 LoRA 热加载能力直接切换推理服务使用的 adapter，
无需重启服务。

前置条件（部署侧）：

- vLLM >= 0.6，启动时开启 ``--enable-lora``；
- 环境变量 ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``（允许运行时加载/卸载 LoRA）。

红线不可绕过：:meth:`VLLMLoRAServer.promote_and_swap` 复用
:meth:`AdapterRegistry.promote` 的校验，未通过评估门的版本一律抛
:class:`evoflow.l2_weight.registry.PromotionDenied`，且不会发出任何热加载请求。

网络层为零依赖 urllib 实现，抽成可注入的 ``http_post`` callable，
签名为 ``http_post(url, payload, headers, timeout) -> dict``，
测试用假实现即可完整覆盖，不真正发请求。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from evoflow.l2_weight.registry import AdapterRegistry
from evoflow.llm.client import LLMError

#: 可注入的 HTTP POST callable 类型：(url, payload, headers, timeout) -> dict
HttpPost = Callable[[str, dict, dict, float], dict]


def _urllib_http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """默认 http_post：标准库 urllib 实现。"""
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"vLLM HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"vLLM 网络错误: {e}") from e
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"vLLM 响应格式异常: {raw[:200]!r}") from e


class VLLMLoRAServer:
    """vLLM OpenAI 兼容服务的 LoRA adapter 热切换客户端。

    :param base_url: vLLM 服务根地址，例如 ``http://localhost:8000``。
    :param api_key: API Key；本地服务可留空。
    :param timeout: 请求超时秒数。
    :param http_post: 可注入的 HTTP POST 实现（测试用假实现替换）。
    """

    LOAD_PATH = "/v1/load_lora_adapter"
    UNLOAD_PATH = "/v1/unload_lora_adapter"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        http_post: HttpPost | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http_post: HttpPost = http_post or _urllib_http_post
        self._active: str | None = None

    @property
    def active_adapter(self) -> str | None:
        """当前热加载中的 adapter 名（本客户端视角）。"""
        return self._active

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return self._http_post(
            f"{self.base_url}{path}", payload, headers, self.timeout
        )

    def load_adapter(self, name: str, path: str | Path) -> dict:
        """热加载一个 LoRA adapter（POST /v1/load_lora_adapter）。

        :param name: adapter 名（后续推理请求用 ``model=name`` 引用）。
        :param path: adapter 权重路径（vLLM 服务侧可见的路径）。
        :return: {"loaded", "path", "response"}。
        """
        if not name or not str(name).strip():
            raise ValueError("adapter name 不能为空")
        body = self._post(
            self.LOAD_PATH, {"lora_name": name, "lora_path": str(path)}
        )
        self._active = name
        return {"loaded": name, "path": str(path), "response": body}

    def unload_adapter(self, name: str) -> dict:
        """卸载一个 LoRA adapter（POST /v1/unload_lora_adapter）。

        :return: {"unloaded", "response"}。
        """
        if not name or not str(name).strip():
            raise ValueError("adapter name 不能为空")
        body = self._post(self.UNLOAD_PATH, {"lora_name": name})
        if self._active == name:
            self._active = None
        return {"unloaded": name, "response": body}

    def promote_and_swap(
        self, registry: AdapterRegistry, version: str | None = None
    ) -> dict:
        """评估门过线即上线：晋升版本并热切换到推理服务。

        红线：``version`` 未通过评估门时，:meth:`AdapterRegistry.promote`
        抛 :class:`~evoflow.l2_weight.registry.PromotionDenied`，本方法
        不发任何热加载请求。

        :param registry: adapter 版本注册表。
        :param version: 要上线的版本；缺省使用 registry 当前已晋升版本。
        :return: {"active", "path", "previous", "unloaded", "response"}。
        :raises ValueError: 未指定 version 且 registry 无已晋升版本。
        """
        if version is None:
            version = registry.current
            if version is None:
                raise ValueError("registry 中尚无已晋升版本，无法热切换")
        else:
            # 复用注册表红线校验：未过评估门在此抛 PromotionDenied。
            registry.promote(version)

        info = registry.info(version)
        adapter_path = info["adapter_path"]

        previous = self._active
        # 先加载新 adapter，成功后再卸载旧的（保证服务始终有可用 adapter）。
        result = self.load_adapter(version, adapter_path)
        unloaded = None
        if previous is not None and previous != version:
            self.unload_adapter(previous)
            unloaded = previous

        return {
            "active": version,
            "path": adapter_path,
            "previous": previous,
            "unloaded": unloaded,
            "response": result["response"],
        }
