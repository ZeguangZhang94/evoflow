"""评委校准测试：kappa 手工值验证、calibrate 结果、分层抽样比例与确定性。"""

import pytest

from evoflow.eval.calibration import (
    CalibrationReport,
    calibrate,
    cohens_kappa,
    sample_for_human,
)

# ---------- cohens_kappa ----------


def test_kappa_known_value():
    """手工值：po=3/4=0.75；pa(y)=3/4, pb(y)=1/2 → pe=1/2；kappa=(0.75-0.5)/0.5=0.5。"""
    a = ["y", "y", "n", "y"]
    b = ["y", "n", "n", "y"]
    assert cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_perfect_agreement():
    assert cohens_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == pytest.approx(1.0)
    # 退化情形：双方全部同一标签，也视为完全一致
    assert cohens_kappa(["y", "y"], ["y", "y"]) == pytest.approx(1.0)


def test_kappa_complete_disagreement():
    """po=0，pe=0.5 → kappa=-1。"""
    assert cohens_kappa(["y", "n"], ["n", "y"]) == pytest.approx(-1.0)


def test_kappa_supports_arbitrary_hashable_labels():
    a = [("pass", 1), ("fail", 0), ("pass", 1)]
    b = [("pass", 1), ("fail", 0), ("fail", 0)]
    # po=2/3；pa(t1)=2/3,pa(t0)=1/3；pb(t1)=1/3,pb(t0)=2/3 → pe=4/9
    # kappa = (2/3 - 4/9) / (1 - 4/9) = (2/9)/(5/9) = 0.4
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_rejects_mismatch_and_empty():
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 2])
    with pytest.raises(ValueError):
        cohens_kappa([], [])


# ---------- calibrate ----------


def test_calibrate_passed():
    judge = [True, True, False, True, False, True]
    human = [True, True, False, True, False, True]
    report = calibrate(judge, human, min_kappa=0.6)
    assert isinstance(report, CalibrationReport)
    assert report.kappa == pytest.approx(1.0)
    assert report.agreement == pytest.approx(1.0)
    assert report.passed is True
    assert report.suggestions == []
    assert report.confusion == {(True, True): 4, (False, False): 2}


def test_calibrate_failed_with_suggestions():
    judge = [True, False, True, False]
    human = [False, True, False, True]  # 完全相反 → kappa=-1
    report = calibrate(judge, human, min_kappa=0.6)
    assert report.passed is False
    assert report.kappa == pytest.approx(-1.0)
    assert report.agreement == 0.0
    assert report.confusion == {
        (True, False): 2,
        (False, True): 2,
    }
    # 未过线必须给出可执行建议
    assert any("评委模型" in s for s in report.suggestions)
    assert any("few-shot" in s for s in report.suggestions)
    assert any("人工复核" in s for s in report.suggestions)
    # to_dict 可 JSON 化（confusion 键转字符串）
    d = report.to_dict()
    assert d["passed"] is False and isinstance(next(iter(d["confusion"])), str)


def test_calibrate_kappa_below_threshold_even_with_high_agreement():
    """一致率 0.85（GPT-4 评委的典型水平）但类别失衡时 kappa 可能不过线。"""
    # 100 条：90 正 10 负；评委把 8 条负判成正、2 条正判成负
    human = [True] * 90 + [False] * 10
    judge = [True] * 88 + [False] * 2 + [True] * 8 + [False] * 2
    report = calibrate(judge, human, min_kappa=0.6)
    # po=0.9；pa(T)=0.96,pb(T)=0.9,pa(F)=0.04,pb(F)=0.1 → pe=0.868
    # kappa=(0.9-0.868)/0.132 ≈ 0.242 < 0.6
    assert report.agreement == pytest.approx(0.9)
    assert report.kappa == pytest.approx((0.9 - 0.868) / (1 - 0.868))
    assert report.passed is False


def test_calibrate_rejects_mismatch_and_empty():
    with pytest.raises(ValueError):
        calibrate([True], [True, False])
    with pytest.raises(ValueError):
        calibrate([], [])


# ---------- sample_for_human ----------


def test_sample_ratio_and_membership():
    items = list(range(100))
    sampled = sample_for_human(items, ratio=0.1)
    assert len(sampled) == 10
    assert all(i in items for i in sampled)
    assert len(set(sampled)) == 10  # 无重复


def test_sample_deterministic_by_seed():
    items = list(range(200))
    a = sample_for_human(items, ratio=0.1, seed=42)
    b = sample_for_human(items, ratio=0.1, seed=42)
    c = sample_for_human(items, ratio=0.1, seed=7)
    assert a == b  # 同 seed 完全可复现
    assert a != c  # 不同 seed 结果不同


def test_sample_stratified_preserves_rare_strata():
    """分层抽样：稀有层（4 条负例）也至少抽到 1 条，且比例大致保持。"""
    items = [{"id": i, "label": "pos"} for i in range(96)] + [
        {"id": 96 + i, "label": "neg"} for i in range(4)
    ]
    sampled = sample_for_human(items, ratio=0.1, seed=42)
    pos = [s for s in sampled if s["label"] == "pos"]
    neg = [s for s in sampled if s["label"] == "neg"]
    assert len(pos) == 10  # round(96*0.1)
    assert len(neg) == 1  # max(1, round(4*0.1)) → 稀有层保底
    assert len(sampled) == 11


def test_sample_without_label_falls_back_to_single_stratum():
    items = [f"task-{i}" for i in range(50)]
    assert len(sample_for_human(items, ratio=0.2)) == 10


def test_sample_rejects_bad_ratio():
    with pytest.raises(ValueError):
        sample_for_human([1, 2, 3], ratio=0)
    with pytest.raises(ValueError):
        sample_for_human([1, 2, 3], ratio=1.5)
