"""黄金集防泄漏测试：精确匹配（含规范化）、char n-gram 近重复、
task/chosen/rejected 三字段全查、assert_no_leakage 红线抛错。"""

import pytest

from evoflow.eval.golden_set import GoldenCase, GoldenSet
from evoflow.eval.leakage import (
    LeakageError,
    LeakageReport,
    assert_no_leakage,
    check_leakage,
)
from evoflow.l2_weight.flywheel import PreferencePair


def _golden(*cases: GoldenCase) -> GoldenSet:
    return GoldenSet(list(cases))


def _pair(task: str, chosen: str = "好答案", rejected: str = "坏答案") -> PreferencePair:
    return PreferencePair(task=task, chosen=chosen, rejected=rejected)


# ---------- 精确匹配 ----------


def test_exact_leak_in_task_detected():
    golden = _golden(GoldenCase(input="如何重置账户密码", expected="设置页操作"))
    pairs = [_pair(task="如何重置账户密码")]
    report = check_leakage(golden, pairs)
    assert isinstance(report, LeakageReport)
    assert report.ok is False
    hit = report.leaked[0]
    assert hit["match"] == "exact"
    assert hit["similarity"] == 1.0
    assert hit["golden_field"] == "input"
    assert hit["train_field"] == "task"
    assert hit["golden_index"] == 0 and hit["train_index"] == 0


def test_exact_match_after_normalization():
    """大小写 / 多余空白差异不影响精确判定（规范化后相等即泄漏）。"""
    golden = _golden(GoldenCase(input="  Reset   PASSWORD ", expected="x"))
    pairs = [_pair(task="reset password")]
    report = check_leakage(golden, pairs)
    assert report.ok is False
    assert report.leaked[0]["match"] == "exact"


def test_golden_expected_also_checked():
    """黄金集的 expected 字段同样受保护。"""
    golden = _golden(GoldenCase(input="普通问题", expected="标准参考答案全文"))
    pairs = [_pair(task="另一个完全不同的任务", chosen="标准参考答案全文")]
    report = check_leakage(golden, pairs)
    assert report.ok is False
    assert report.leaked[0]["golden_field"] == "expected"
    assert report.leaked[0]["train_field"] == "chosen"


def test_rejected_field_also_checked():
    """rejected 字段里的泄漏同样算数——三字段无一豁免。"""
    golden = _golden(GoldenCase(input="黄金题目原文", expected="y"))
    pairs = [_pair(task="无关任务", chosen="无关好答", rejected="黄金题目原文")]
    report = check_leakage(golden, pairs)
    assert report.ok is False
    assert report.leaked[0]["train_field"] == "rejected"


# ---------- 近重复 ----------


def test_near_duplicate_detected():
    """仅增删个别字符的改写（Jaccard > 0.8）必须被近重复检测抓住。"""
    golden = _golden(GoldenCase(input="如何重置我的账户密码", expected="x"))
    pairs = [_pair(task="如何重置我的账户密码。")]  # 只多了一个句号
    report = check_leakage(golden, pairs)
    assert report.ok is False
    hit = report.leaked[0]
    assert hit["match"] == "near"
    assert 0.8 < hit["similarity"] < 1.0


def test_distinct_texts_pass():
    golden = _golden(
        GoldenCase(input="如何重置账户密码", expected="进入设置页操作"),
        GoldenCase(input="退款流程是什么", expected="在订单页申请退款"),
    )
    pairs = [
        _pair(task="推荐几本机器学习入门书", chosen="《统计学习方法》", rejected="随便看看"),
        _pair(task="今天天气怎么样", chosen="晴天", rejected="不知道"),
    ]
    report = check_leakage(golden, pairs)
    assert report.ok is True
    assert report.leaked == []


def test_empty_train_pairs_pass():
    golden = _golden(GoldenCase(input="q", expected="a"))
    assert check_leakage(golden, []).ok is True


def test_threshold_boundary_not_leaking():
    """相似度恰好在阈值以下时不误报。"""
    golden = _golden(GoldenCase(input="abcdefgh", expected="x"))
    # "abcXYZgh" 与 "abcdefgh" 的 trigram 仅共享 "abc" → Jaccard=1/11≈0.09
    pairs = [_pair(task="abcXYZgh")]
    assert check_leakage(golden, pairs, threshold=0.8).ok is True


# ---------- 红线 assert ----------


def test_assert_no_leakage_raises_with_report():
    golden = _golden(GoldenCase(input="机密黄金题", expected="机密答案"))
    pairs = [_pair(task="机密黄金题"), _pair(task="正常任务")]
    with pytest.raises(LeakageError) as exc_info:
        assert_no_leakage(golden, pairs)
    err = exc_info.value
    assert "泄漏" in str(err)
    assert err.report.ok is False
    assert len(err.report.leaked) == 1  # 只有第一条 pair 泄漏


def test_assert_no_leakage_returns_report_when_clean():
    golden = _golden(GoldenCase(input="如何开发票", expected="财务系统申请"))
    pairs = [_pair(task="怎么做俯卧撑", chosen="标准动作", rejected="错误姿势")]
    report = assert_no_leakage(golden, pairs)
    assert report.ok is True


def test_dict_shaped_pairs_supported():
    """训练对为 dict（jsonl 反序列化形态）时同样可查。"""
    golden = _golden(GoldenCase(input="黄金原题", expected="x"))
    pairs = [{"task": "黄金原题", "chosen": "c", "rejected": "r"}]
    report = check_leakage(golden, pairs)
    assert report.ok is False
    assert report.leaked[0]["match"] == "exact"
