from typing import Dict, List, Optional, Tuple
import numpy as np

from metrics import compute_multilabel_metrics, to_dict as metrics_to_dict
from calibration import compute_multilabel_calibration, to_dict as cal_to_dict
from metrics import validate_multilabel_inputs


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.nanpercentile(values, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_multilabel(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    *,
    label_names: Optional[List[str]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    n_bins: int = 15,
) -> Dict:
    Y_true, Y_prob, label_names = validate_multilabel_inputs(Y_true, Y_prob, label_names)
    rng = np.random.default_rng(seed)
    n = Y_true.shape[0]

    point_metrics = metrics_to_dict(compute_multilabel_metrics(Y_true, Y_prob, label_names))
    point_cal = cal_to_dict(compute_multilabel_calibration(Y_true, Y_prob, label_names, n_bins=n_bins, keep_reliability=False))

    # We'll collect bootstrap replicates for:
    # - global scalars: auroc_macro, auprc_macro, brier_macro, logloss_macro, auroc_micro, auprc_micro
    # - per-label scalars for auroc/auprc/brier/logloss and calibration brier/ece/mce/slope/intercept
    scalar_keys = [
        "auroc_macro",
        "auprc_macro",
        "brier_macro",
        "logloss_macro",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "auroc_micro",
        "auprc_micro",
        "f1_micro",
        "precision_micro",
        "recall_micro",
        "ece_macro",
    ]

    boot_scalars = {k: np.full(n_boot, np.nan, dtype=float) for k in scalar_keys}
    boot_perlabel = {
        "auroc": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "auprc": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "brier": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "logloss": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "cal_brier": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "ece": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "mce": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "cal_intercept": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
        "cal_slope": {name: np.full(n_boot, np.nan, dtype=float) for name in label_names},
    }
    print("Bootstrap starting...")

    for b in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yt = Y_true[idx, :]
        yp = Y_prob[idx, :]

        m = metrics_to_dict(compute_multilabel_metrics(yt, yp, label_names))
        c = cal_to_dict(compute_multilabel_calibration(yt, yp, label_names, n_bins=n_bins, keep_reliability=False))

        for k in scalar_keys:
            if k == "ece_macro":
                # Mean ECE for all classe
                ece_values = [c["ece"][name] for name in label_names]
                boot_scalars[k][b] = float(np.mean(ece_values))
            else:
                boot_scalars[k][b] = m[k]

        for name in label_names:
            boot_perlabel["auroc"][name][b] = m["auroc"][name]
            boot_perlabel["auprc"][name][b] = m["auprc"][name]
            boot_perlabel["brier"][name][b] = m["brier"][name]
            boot_perlabel["logloss"][name][b] = m["logloss"][name]

            boot_perlabel["cal_brier"][name][b] = c["brier"][name]
            boot_perlabel["ece"][name][b] = c["ece"][name]
            boot_perlabel["mce"][name][b] = c["mce"][name]
            boot_perlabel["cal_intercept"][name][b] = c["cal_intercept"][name]
            boot_perlabel["cal_slope"][name][b] = c["cal_slope"][name]

        if b % 50 == 0:
            print(f"Completed {b} / {n_boot} bootstrap replicates")

    def ci_dict_from_array(arr: np.ndarray) -> Dict[str, float]:
        if np.all(np.isnan(arr)):
            return {"lo": float("nan"), "hi": float("nan")}
        lo, hi = percentile_ci(arr, alpha=alpha)
        return {"lo": lo, "hi": hi}

    ci = {"scalars": {}, "per_label": {}}
    for k in scalar_keys:
        ci["scalars"][k] = ci_dict_from_array(boot_scalars[k])

    # per label CI
    for group, per_label_map in boot_perlabel.items():
        ci["per_label"][group] = {}
        for name, arr in per_label_map.items():
            ci["per_label"][group][name] = ci_dict_from_array(arr)

    return {
        "point": {
            "metrics": point_metrics,
            "calibration": point_cal,
        },
        "ci": ci,
        "meta": {
            "n": int(n),
            "n_labels": int(Y_true.shape[1]),
            "label_names": label_names,
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "seed": int(seed),
            "n_bins": int(n_bins),
            "bootstrap_unit": "rows",
            "ci_method": "percentile",
        },
    }
