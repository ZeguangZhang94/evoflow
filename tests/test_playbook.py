"""L0 Playbook 测试：curate 去重合并、render token 预算、save/load 往返。"""

from evoflow.l0_context.playbook import DeltaEntry, Playbook


def _entry(content: str, tags=None, hits: int = 0) -> DeltaEntry:
    return DeltaEntry(id=f"id-{content[:6]}", content=content, tags=tags or [], hits=hits)


def test_curate_dedup_merges_tags_and_hits():
    pb = Playbook()
    pb.curate([_entry("回答要给具体入口", tags=["a"], hits=1)])
    added = pb.curate([_entry("  回答要给具体入口 ", tags=["b"], hits=2)])
    assert added == []  # 规范化后内容相同，不新增
    assert len(pb) == 1
    entry = pb.entries[0]
    assert entry.tags == ["a", "b"]
    assert entry.hits == 3  # hits 累加


def test_curate_adds_new_entries():
    pb = Playbook()
    added = pb.curate([_entry("规则一"), _entry("规则二"), _entry("规则一")])
    assert len(added) == 2
    assert len(pb) == 2


def test_render_respects_max_tokens():
    pb = Playbook()
    pb.curate([_entry("很长的规则" * 100)])
    out = pb.render(max_tokens=10)
    # 预算极小：只保留标题行，正文条目被截断
    assert "Playbook" in out
    assert "很长的规则" not in out


def test_render_orders_by_hits_desc():
    pb = Playbook()
    pb.curate([_entry("冷门规则", hits=0), _entry("高频规则", hits=9)])
    out = pb.render(max_tokens=10000)
    assert out.index("高频规则") < out.index("冷门规则")


def test_save_load_roundtrip(tmp_path):
    pb = Playbook([_entry("规则甲", tags=["x"], hits=5)])
    path = tmp_path / "playbook.json"
    pb.save(path)
    loaded = Playbook.load(path)
    assert len(loaded) == 1
    e = loaded.entries[0]
    assert e.content == "规则甲" and e.tags == ["x"] and e.hits == 5


def test_load_missing_file_returns_empty(tmp_path):
    assert len(Playbook.load(tmp_path / "nope.json")) == 0
