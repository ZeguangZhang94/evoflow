"""评估层：冻结黄金集与隔离评委（评估门的两大构件），
外加评委校准与黄金集防泄漏两条红线的检查工具。"""

from evoflow.eval.calibration import (
    CalibrationReport,
    calibrate,
    cohens_kappa,
    sample_for_human,
)
from evoflow.eval.leakage import (
    LeakageError,
    LeakageReport,
    assert_no_leakage,
    check_leakage,
)

__all__ = [
    "CalibrationReport",
    "LeakageError",
    "LeakageReport",
    "assert_no_leakage",
    "calibrate",
    "check_leakage",
    "cohens_kappa",
    "sample_for_human",
]
