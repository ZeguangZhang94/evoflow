"""共享 LLM 客户端 —— 零依赖 OpenAI 兼容协议实现。

设计原则（与 EvoFlow 核心承诺一致）：
- 本模块**纯标准库**（urllib），不依赖 openai 包；
- 兼容所有 OpenAI Chat Completions 协议的端点：
  OpenAI、DeepSeek、阿里云百炼（DashScope 兼容模式）、vLLM/Ollama 本地服务等；
- 需要 openai SDK 高级特性的用户可自行实现 :class:`LLMClient` 协议注入。

用法::

    from evoflow.llm import OpenAICompatClient
    client = OpenAICompatClient(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-...",
        model="qwen-flash",
    )
    text = client.chat([{"role": "user", "content": "你好"}])
"""

from .client import LLMClient, LLMError, OpenAICompatClient, ScriptedLLM

__all__ = ["LLMClient", "LLMError", "OpenAICompatClient", "ScriptedLLM"]
