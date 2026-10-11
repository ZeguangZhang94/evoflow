"""零依赖 OpenAI 兼容 Chat Completions 客户端。

仅使用标准库 urllib 实现，供 EvoFlow 各进化模块（Reflector、LLMJudge、
GEPA、知识层等）共享，避免每个模块各写一套 HTTP 逻辑。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(RuntimeError):
    """LLM 调用失败（网络错误、HTTP 非 200、响应格式异常）。"""


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端协议：任何可注入 EvoFlow 的模型调用方都应满足。"""

    def chat(self, messages: Sequence[Message], **kwargs) -> str:
        """发送一组消息，返回助手回复文本。"""
        ...


class OpenAICompatClient:
    """OpenAI Chat Completions 兼容客户端（纯标准库）。

    :param base_url: 端点根地址，例如
        ``https://api.openai.com/v1``、
        ``https://api.deepseek.com/v1``、
        ``https://dashscope.aliyuncs.com/compatible-mode/v1``（阿里云百炼）、
        ``http://localhost:8000/v1``（vLLM/Ollama 本地服务）。
    :param api_key: API Key；本地服务可传任意非空字符串。
    :param model: 默认模型名。
    :param timeout: 请求超时秒数。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: Sequence[Message], **kwargs) -> str:
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": list(messages),
            **kwargs,
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"LLM HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"LLM 网络错误: {e}") from e
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"LLM 响应格式异常: {body!r}") from e


class ScriptedLLM:
    """测试/演示用假 LLM：按队列依次返回预置答案，用完后返回默认答案。

    同时记录收到的全部消息，便于断言提示词内容。

    :param answers: 依次返回的答案队列。
    :param default: 队列耗尽后的默认返回。
    """

    def __init__(self, answers: Sequence[str] = (), default: str = "") -> None:
        self._answers = list(answers)
        self.default = default
        self.calls: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], **kwargs) -> str:
        self.calls.append(list(messages))
        if self._answers:
            return self._answers.pop(0)
        return self.default
