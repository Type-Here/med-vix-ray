from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import validate_multilabel_inputs


def expected_calibration_error_1d(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    mean_pred = []
    frac_pos = []
    counts = []
    n = y_true.size

    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        counts.append(cnt)
        if cnt == 0:
            mean_pred.append(float("nan"))
            frac_pos.append(float("nan"))
            continue
        mp = float(y_prob[mask].mean())
        fp = float(y_true[mask].mean())
        mean_pred.append(mp)
        frac_pos.append(fp)
        gap = abs(fp - mp)
        ece += (cnt / n) * gap
        mce = max(mce, gap)

    return float(ece), float(mce), {
        "edges": edges.tolist(),
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "counts": counts,
    }


def calibration_slope_intercept_1d(y_true: np.ndarray, y_prob: np.ndarray):
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    # If y_true has single class, slope/intercept aren't meaningful.
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit_p, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


@dataclass
class MultilabelCalibrationResults:
    n: int
    n_labels: int
    n_bins: int

    brier: Dict[str, float]
    ece: Dict[str, float]
    mce: Dict[str, float]
    cal_intercept: Dict[str, float]
    cal_slope: Dict[str, float]

    # optional reliability data per label (for plots)
    reliability: Dict[str, Dict]


def compute_multilabel_calibration(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    label_names: Optional[List[str]] = None,
    n_bins: int = 15,
    keep_reliability: bool = True,
) -> MultilabelCalibrationResults:
    Y_true, Y_prob, label_names = validate_multilabel_inputs(Y_true, Y_prob, label_names)
    n, L = Y_true.shape

    brier = {}
    ece = {}
    mce = {}
    cint = {}
    cslope = {}
    rel = {}

    for j, name in enumerate(label_names):
        yt = Y_true[:, j]
        yp = Y_prob[:, j]

        brier[name] = float(np.mean((yp - yt) ** 2))
        ece_val, mce_val, stats = expected_calibration_error_1d(yt, yp, n_bins=n_bins)
        ece[name] = ece_val
        mce[name] = mce_val

        a, b = calibration_slope_intercept_1d(yt, yp)
        cint[name] = a
        cslope[name] = b

        if keep_reliability:
            rel[name] = stats

    return MultilabelCalibrationResults(
        n=int(n),
        n_labels=int(L),
        n_bins=int(n_bins),
        brier=brier,
        ece=ece,
        mce=mce,
        cal_intercept=cint,
        cal_slope=cslope,
        reliability=rel if keep_reliability else {},
    )


def to_dict(res: MultilabelCalibrationResults) -> Dict:
    return asdict(res)
