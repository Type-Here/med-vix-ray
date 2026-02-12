from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             precision_score, recall_score, f1_score)


def validate_multilabel_inputs(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    Y_true = np.asarray(Y_true).astype(int)
    Y_prob = np.asarray(Y_prob).astype(float)

    if Y_true.ndim != 2 or Y_prob.ndim != 2:
        raise ValueError("Y_true and Y_prob must be 2D arrays: (n_samples, n_labels)")
    if Y_true.shape != Y_prob.shape:
        raise ValueError(f"Shape mismatch: {Y_true.shape} vs {Y_prob.shape}")

    if not np.all((Y_true == 0) | (Y_true == 1)):
        raise ValueError("Y_true must be binary {0,1}")

    if np.any(~np.isfinite(Y_prob)):
        raise ValueError("Y_prob contains non-finite values")

    Y_prob = np.clip(Y_prob, 1e-15, 1 - 1e-15)

    n_labels = Y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError("label_names length must match n_labels")

    return Y_true, Y_prob, label_names


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")


@dataclass
class MultilabelMetricResults:
    n: int
    n_labels: int

    # macro over labels
    auroc_macro: float
    auprc_macro: float
    brier_macro: float
    logloss_macro: float
    f1_macro: float
    precision_macro: float
    recall_macro: float

    # micro over all label decisions (flatten)
    auroc_micro: float
    auprc_micro: float
    f1_micro: float
    precision_micro: float
    recall_micro: float

    # per label dicts
    prevalence: Dict[str, float]
    auroc: Dict[str, float]
    auprc: Dict[str, float]
    brier: Dict[str, float]
    logloss: Dict[str, float]


def compute_multilabel_metrics(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> MultilabelMetricResults:
    Y_true, Y_prob, label_names = validate_multilabel_inputs(Y_true, Y_prob, label_names)
    n, L = Y_true.shape

    prev = {}
    auroc = {}
    auprc = {}
    brier = {}
    ll = {}

    auroc_arr = np.full(L, np.nan, dtype=float)
    auprc_arr = np.full(L, np.nan, dtype=float)
    brier_arr = np.full(L, np.nan, dtype=float)
    ll_arr = np.full(L, np.nan, dtype=float)

    for j, name in enumerate(label_names):
        yt = Y_true[:, j]
        yp = Y_prob[:, j]

        prev[name] = float(yt.mean())

        # AUROC/AUPRC undefined if only one class present
        if len(np.unique(yt)) == 2:
            auroc_val = float(roc_auc_score(yt, yp))
            auprc_val = float(average_precision_score(yt, yp))
        else:
            auroc_val = float("nan")
            auprc_val = float("nan")

        # Brier and logloss are still computable with single-class, but logloss needs both labels in sklearn
        brier_val = float(np.mean((yp - yt) ** 2))

        # log_loss: force labels [0,1] to avoid error when one class missing
        ll_val = float(log_loss(yt, yp, labels=[0, 1]))

        auroc[name] = auroc_val
        auprc[name] = auprc_val
        brier[name] = brier_val
        ll[name] = ll_val

        auroc_arr[j] = auroc_val
        auprc_arr[j] = auprc_val
        brier_arr[j] = brier_val
        ll_arr[j] = ll_val

    auroc_macro = _nanmean(auroc_arr)
    auprc_macro = _nanmean(auprc_arr)
    brier_macro = float(np.mean(brier_arr))  # always finite
    logloss_macro = float(np.mean(ll_arr))   # always finite

    # For F1/precision/recall we need thresholded predictions; use 0.5
    y_pred = (Y_prob >= 0.5).astype(int)

    # macro: average across labels (handles missing classes via zero_division)
    try:
        precision_macro = float(precision_score(Y_true, y_pred, average="macro", zero_division=0))
        recall_macro = float(recall_score(Y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(Y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        precision_macro = float("nan")
        recall_macro = float("nan")
        f1_macro = float("nan")

    # micro: global (equivalent to flatten)
    try:
        precision_micro = float(precision_score(Y_true, y_pred, average="micro", zero_division=0))
        recall_micro = float(recall_score(Y_true, y_pred, average="micro", zero_division=0))
        f1_micro = float(f1_score(Y_true, y_pred, average="micro", zero_division=0))
    except Exception:
        precision_micro = float("nan")
        recall_micro = float("nan")
        f1_micro = float("nan")

    # micro: flatten all label decisions
    yt_flat = Y_true.reshape(-1)
    yp_flat = Y_prob.reshape(-1)
    if len(np.unique(yt_flat)) == 2:
        auroc_micro = float(roc_auc_score(yt_flat, yp_flat))
        auprc_micro = float(average_precision_score(yt_flat, yp_flat))
    else:
        auroc_micro = float("nan")
        auprc_micro = float("nan")

    return MultilabelMetricResults(
        n=int(n),
        n_labels=int(L),
        auroc_macro=auroc_macro,
        auprc_macro=auprc_macro,
        brier_macro=brier_macro,
        logloss_macro=logloss_macro,
        f1_macro=f1_macro,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        auroc_micro=auroc_micro,
        auprc_micro=auprc_micro,
        f1_micro=f1_micro,
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        prevalence=prev,
        auroc=auroc,
        auprc=auprc,
        brier=brier,
        logloss=ll,
    )


def to_dict(res: MultilabelMetricResults) -> Dict:
    return asdict(res)
