"""L1 经验/记忆库 —— 向量存储（嵌入检索底座）。

提供 :class:`VectorStore` 协议与本地零依赖实现 :class:`JsonlVectorStore`。
默认嵌入为 dependency-free 的字符 3-gram hashing 向量 + 余弦相似度；
``embed_fn`` 可插拔（如安装 ``evoflow[vec]`` 或接入外部 embedding 服务）。

注意：hashing 必须使用确定性哈希（md5），不能用内置 ``hash()``——后者
按进程加盐，会导致落盘向量跨进程不可复现。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

EmbedFn = Callable[[str], list[float]]


def hashing_embed(text: str, dim: int = 512) -> list[float]:
    """零依赖字符 n-gram hashing 嵌入（L2 归一化）。

    以字符 3-gram 为主，叠加 1/2-gram 提升中文等无空格语言的召回
    （纯 3-gram 在跨字序查询下容易零命中）。

    :param text: 输入文本。
    :param dim: 向量维度（hashing 桶数）。
    """
    vec = [0.0] * dim
    t = text.lower()
    if len(t) < 3:
        t = t + " " + t  # 保证短文本也有 3-gram
    for n, weight in ((1, 0.5), (2, 1.0), (3, 1.0)):
        for i in range(len(t) - n + 1):
            gram = t[i : i + n]
            bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % dim
            vec[bucket] += weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（假定输入可能未归一化，做防御式归一）。"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore(Protocol):
    """向量存储协议：add / search / save / load。"""

    def add(self, item_id: str, text: str, payload: dict) -> None:
        """加入一条记录（text 用于嵌入，payload 原样存取）。"""
        ...

    def search(self, query: str, k: int = 1) -> list[tuple[dict, float]]:
        """按相似度检索，返回 [(payload, score)] 降序列表，最多 k 条。"""
        ...

    def save(self, path: str | Path) -> None:
        """落盘。"""
        ...

    def load(self, path: str | Path) -> None:
        """加载。"""
        ...


class JsonlVectorStore:
    """本地 jsonl 向量存储实现。

    :param embed_fn: 可插拔嵌入函数；缺省使用 :func:`hashing_embed`。
    :param dim: 默认嵌入的维度（仅对默认 embed_fn 生效）。
    """

    def __init__(self, embed_fn: EmbedFn | None = None, dim: int = 512) -> None:
        self.embed_fn: EmbedFn = embed_fn or (lambda t: hashing_embed(t, dim))
        self._records: list[dict] = []  # {id, text, payload, vector}

    def add(self, item_id: str, text: str, payload: dict) -> None:
        self._records.append(
            {
                "id": item_id,
                "text": text,
                "payload": payload,
                "vector": self.embed_fn(text),
            }
        )

    def search(self, query: str, k: int = 1) -> list[tuple[dict, float]]:
        qv = self.embed_fn(query)
        scored = [(r["payload"], cosine(qv, r["vector"])) for r in self._records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(0, k)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)
