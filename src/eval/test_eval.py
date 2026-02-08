from metrics import compute_multilabel_metrics
from calibration import compute_multilabel_calibration
from bootstrap import bootstrap_multilabel


def test_multilabel_metrics_shapes():
    rng = np.random.default_rng(0)
    n, L = 200, 14
    Y_true = rng.integers(0, 2, size=(n, L))
    Y_prob = rng.random(size=(n, L))
    names = [f"l{i}" for i in range(L)]

    res = compute_multilabel_metrics(Y_true, Y_prob, names)
    assert res.n == n and res.n_labels == L
    assert len(res.auroc) == L
    assert np.isfinite(res.brier_macro)


def test_multilabel_calibration_runs():
    rng = np.random.default_rng(1)
    n, L = 300, 14
    Y_true = rng.integers(0, 2, size=(n, L))
    Y_prob = rng.random(size=(n, L))
    names = [f"l{i}" for i in range(L)]

    cal = compute_multilabel_calibration(Y_true, Y_prob, names, n_bins=10, keep_reliability=False)
    assert len(cal.ece) == L
    assert len(cal.cal_slope) == L


def test_bootstrap_returns_ci():
    rng = np.random.default_rng(2)
    n, L = 250, 14
    Y_true = rng.integers(0, 2, size=(n, L))
    Y_prob = rng.random(size=(n, L))
    names = [f"l{i}" for i in range(L)]

    out = bootstrap_multilabel(Y_true, Y_prob, label_names=names, n_boot=100, seed=0)
    assert "point" in out and "ci" in out
    assert "scalars" in out["ci"]
    assert "auroc_macro" in out["ci"]["scalars"]
    assert "per_label" in out["ci"]
    assert "auroc" in out["ci"]["per_label"]
    assert "l0" in out["ci"]["per_label"]["auroc"]
