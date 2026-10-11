"""vLLM LoRA 热切换测试：假 http_post 覆盖 load/unload/promote_and_swap 与红线。"""

import pytest

from evoflow.core.gate import GateDecision, GateResult
from evoflow.l2_weight.registry import AdapterRegistry, PromotionDenied
from evoflow.l2_weight.serving import VLLMLoRAServer
from evoflow.llm.client import LLMError

PASSED = GateResult(passed=True, pass_rate=0.95, decision=GateDecision.PROMOTE)


class FakeHTTP:
    """假 http_post：记录全部调用，返回固定响应。"""

    def __init__(self, response=None, error=None):
        self.calls: list[tuple[str, dict, dict, float]] = []
        self.response = response if response is not None else {"status": "ok"}
        self.error = error

    def __call__(self, url: str, payload: dict, headers: dict, timeout: float) -> dict:
        self.calls.append((url, payload, headers, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def _server(http: FakeHTTP) -> VLLMLoRAServer:
    return VLLMLoRAServer("http://localhost:8000/", http_post=http)


class TestLoadUnload:
    def test_load_adapter_posts_correct_payload(self):
        http = FakeHTTP()
        server = _server(http)
        result = server.load_adapter("v0001", "/data/adapters/v0001")
        assert result["loaded"] == "v0001"
        assert server.active_adapter == "v0001"
        url, payload, headers, timeout = http.calls[0]
        assert url == "http://localhost:8000/v1/load_lora_adapter"
        assert payload == {"lora_name": "v0001", "lora_path": "/data/adapters/v0001"}
        assert headers == {} and timeout == 30.0

    def test_unload_adapter_posts_and_clears_active(self):
        http = FakeHTTP()
        server = _server(http)
        server.load_adapter("v0001", "/a")
        result = server.unload_adapter("v0001")
        assert result["unloaded"] == "v0001"
        assert server.active_adapter is None
        url, payload, _, _ = http.calls[-1]
        assert url == "http://localhost:8000/v1/unload_lora_adapter"
        assert payload == {"lora_name": "v0001"}

    def test_api_key_sent_as_bearer(self):
        http = FakeHTTP()
        server = VLLMLoRAServer("http://x:8000", api_key="k", http_post=http)
        server.load_adapter("v1", "/a")
        assert http.calls[0][2] == {"Authorization": "Bearer k"}

    def test_empty_name_rejected(self):
        server = _server(FakeHTTP())
        with pytest.raises(ValueError):
            server.load_adapter("", "/a")
        with pytest.raises(ValueError):
            server.unload_adapter("  ")

    def test_http_error_propagates_as_llm_error(self):
        http = FakeHTTP(error=LLMError("vLLM HTTP 500: boom"))
        server = _server(http)
        with pytest.raises(LLMError, match="500"):
            server.load_adapter("v1", "/a")
        assert server.active_adapter is None  # 失败不改变活动指针


class TestPromoteAndSwap:
    def test_swaps_to_promoted_version(self, tmp_path):
        registry = AdapterRegistry(tmp_path)
        v1 = registry.register(tmp_path / "a1", metrics={}, gate_result=PASSED)
        v2 = registry.register(tmp_path / "a2", metrics={}, gate_result=PASSED)
        registry.promote(v1)

        http = FakeHTTP()
        server = _server(http)
        server.load_adapter(v1, str(tmp_path / "a1"))  # 当前线上跑 v1

        result = server.promote_and_swap(registry, v2)
        assert result["active"] == v2
        assert result["previous"] == v1
        assert result["unloaded"] == v1
        assert server.active_adapter == v2
        # 先 load 新，后 unload 旧（服务始终有可用 adapter）
        assert http.calls[-2][0].endswith("/v1/load_lora_adapter")
        assert http.calls[-1][0].endswith("/v1/unload_lora_adapter")

    def test_default_uses_registry_current(self, tmp_path):
        registry = AdapterRegistry(tmp_path)
        v1 = registry.register(tmp_path / "a1", metrics={}, gate_result=PASSED)
        registry.promote(v1)
        http = FakeHTTP()
        server = _server(http)
        result = server.promote_and_swap(registry)
        assert result["active"] == v1
        assert result["unloaded"] is None  # 无旧版本，不卸载

    def test_unpromoted_version_denied_and_no_http(self, tmp_path):
        """红线：未过评估门的版本禁止热切换，且不发任何请求。"""
        registry = AdapterRegistry(tmp_path)
        v1 = registry.register(tmp_path / "a1", metrics={})  # 无 gate 结果
        http = FakeHTTP()
        server = _server(http)
        with pytest.raises(PromotionDenied):
            server.promote_and_swap(registry, v1)
        assert http.calls == []
        assert server.active_adapter is None

    def test_no_version_and_no_current_rejected(self, tmp_path):
        registry = AdapterRegistry(tmp_path)
        server = _server(FakeHTTP())
        with pytest.raises(ValueError, match="已晋升"):
            server.promote_and_swap(registry)
