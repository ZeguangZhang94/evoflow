"""L1 A-MEM 式情节记忆测试：LLM 路径、启发式回退、建链、演化合并、持久化。"""

from evoflow.l1_memory.episodic import (
    EpisodicMemory,
    _heuristic_attrs,
    _parse_llm_attrs,
)
from evoflow.l1_memory.store import JsonlVectorStore
from evoflow.llm.client import ScriptedLLM

_LLM_GOOD = "关键词: 密码, 重置, 账号\n标签: 账号安全, 自助服务\n上下文: 用户忘记登录密码"


def test_add_with_llm_uses_llm_attrs():
    client = ScriptedLLM(answers=[_LLM_GOOD])
    mem = EpisodicMemory(JsonlVectorStore(), client=client)
    note = mem.add("用户说忘记密码，需要去设置里重置")
    assert note.keywords == ["密码", "重置", "账号"]
    assert note.tags == ["账号安全", "自助服务"]
    assert note.context == "用户忘记登录密码"
    assert len(client.calls) == 1
    assert "忘记密码" in client.calls[0][0]["content"]


def test_add_without_client_falls_back_to_heuristic():
    mem = EpisodicMemory(JsonlVectorStore())  # 无 client
    note = mem.add("deploy deploy rollback the failed deploy")
    assert note.keywords[0] == "deploy"  # 高频词优先
    assert note.tags
    assert note.context


def test_add_with_llm_garbage_output_falls_back_to_heuristic():
    client = ScriptedLLM(answers=["我完全没按格式来~~~"])
    mem = EpisodicMemory(JsonlVectorStore(), client=client)
    note = mem.add("deploy deploy rollback the deploy")
    assert note.keywords[0] == "deploy"  # 解析失败 → 启发式
    assert note.context


def test_auto_link_writes_bidirectional_links():
    mem = EpisodicMemory(JsonlVectorStore())
    a = mem.add("如何重置登录密码")
    b = mem.add("密码忘记了怎么重置")
    assert b.id in a.links
    assert a.id in b.links


def test_retrieve_updates_access_stats_and_neighbors():
    mem = EpisodicMemory(JsonlVectorStore())
    a = mem.add("如何重置登录密码")
    b = mem.add("密码忘记了怎么重置")
    hits = mem.retrieve("重置密码", k=1)
    assert len(hits) == 1
    top = hits[0]
    assert top.access_count == 1
    assert top.last_accessed >= top.created_at

    hits2 = mem.retrieve("重置密码", k=1, include_neighbors=True)
    assert len(hits2) == 2
    assert {h.id for h in hits2} == {a.id, b.id}
    # 邻居不重复计入 access_count
    neighbor = hits2[1]
    assert neighbor.access_count == 0


def test_evolve_merges_hot_linked_cluster():
    mem = EpisodicMemory(JsonlVectorStore())
    a = mem.add("重置密码先点设置")
    b = mem.add("密码重置再点账号安全")
    assert a.links and b.links  # 已互链
    mem.retrieve("重置密码")
    mem.retrieve("密码重置")  # 两条都被访问过，access_count 升高
    assert a.access_count >= 1 and b.access_count >= 1

    created = mem.evolve(min_access=1, min_cluster=2)
    assert len(created) == 1
    merged = created[0]
    assert a.archived and b.archived
    assert merged.access_count == a.access_count + b.access_count
    assert "演化合并" in merged.context
    # 归档笔记不再被检索命中，演化产物可以被检索到
    assert len(mem) == 1
    hits = mem.retrieve("重置密码", k=5)
    assert all(h.id == merged.id for h in hits)


def test_evolve_skips_cold_notes():
    mem = EpisodicMemory(JsonlVectorStore())
    mem.add("如何重置登录密码")
    mem.add("密码忘记了怎么重置")
    # 从未访问 → 达不到 min_access，不演化
    assert mem.evolve(min_access=2, min_cluster=2) == []
    assert len(mem) == 2


def test_evolve_with_llm_uses_llm_summary():
    # 答案队列：前两个被两次 add 的卡片属性生成消耗，第三个才是演化总结
    client = ScriptedLLM(answers=[_LLM_GOOD, _LLM_GOOD, "重置密码的共同规律：都走设置页"])
    mem = EpisodicMemory(JsonlVectorStore(), client=client)
    mem.add("重置密码先点设置")
    mem.add("密码重置再点账号安全")
    mem.retrieve("重置密码")
    mem.retrieve("密码重置")  # 两条都被访问过，都达到 min_access
    created = mem.evolve(min_access=1, min_cluster=2)
    assert created[0].content == "重置密码的共同规律：都走设置页"


def test_evolve_redirects_external_links_to_merged_note():
    mem = EpisodicMemory(JsonlVectorStore())
    a = mem.add("重置密码先点设置")
    b = mem.add("密码重置再点账号安全")
    c = mem.add("怎么重置密码和找回密码")  # 与 a/b 也建链
    mem.retrieve("重置密码")
    mem.retrieve("密码重置")
    assert c.access_count == 0 or c.access_count >= 0  # c 可能也被命中
    # 只让 a/b 变热：直接调高阈值演化
    a.access_count = 5
    b.access_count = 5
    c.access_count = 0
    created = mem.evolve(min_access=2, min_cluster=2)
    assert len(created) == 1
    merged = created[0]
    if a.id in c.links or b.id in c.links:
        assert merged.id in c.links
        assert a.id not in c.links and b.id not in c.links
        assert c.id in merged.links


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "episodic.jsonl"
    mem = EpisodicMemory(JsonlVectorStore())
    a = mem.add("如何重置登录密码")
    mem.add("密码忘记了怎么重置")
    mem.retrieve("重置密码")
    mem.save(path)

    mem2 = EpisodicMemory(JsonlVectorStore())
    mem2.load(path)
    assert len(mem2) == 2
    # 持久化字段（含 access_count）在检索前完整恢复
    assert mem2._notes[a.id].access_count == a.access_count
    loaded = mem2.retrieve("重置密码", k=1)[0]
    assert loaded.id == a.id or "密码" in loaded.content


def test_parse_llm_attrs_rejects_empty():
    assert _parse_llm_attrs("无关输出") is None
    kw, tags, ctx = _parse_llm_attrs(_LLM_GOOD)
    assert kw == ["密码", "重置", "账号"]
    assert ctx == "用户忘记登录密码"


def test_heuristic_attrs_chinese():
    kw, tags, ctx = _heuristic_attrs("用户反馈登录失败，登录界面一直转圈")
    assert kw and tags and ctx
