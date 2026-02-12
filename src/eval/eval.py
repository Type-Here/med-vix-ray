import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from settings import MIMIC_LABELS
from bootstrap import bootstrap_multilabel
from src.med_vix_ray import SwinMIMICGraphClassifier
from src.fine_tuned_model import SwinMIMICClassifier


@torch.no_grad()
def collect_probs_and_labels(
    model: Union[SwinMIMICGraphClassifier,SwinMIMICClassifier],
    loader: torch.utils.data.DataLoader,
    device: str,
    use_nudger: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true_list = []
    y_prob_list = []

    if isinstance(model, SwinMIMICGraphClassifier):
        model.is_using_nudger = use_nudger

    count = 0
    length = len(loader)
    print("Collecting probs and labels")
    for batch in loader:
        # batch is tuple: (image, labels) OR (image, labels, study_id)
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].detach().cpu().numpy()  # (B, 14)

        logits = model(x)  # expected (B, 14)
        if logits.ndim != 2:
            raise ValueError(f"Model output must be 2D (B, L). Got shape: {tuple(logits.shape)}")

        prob = torch.sigmoid(logits).detach().cpu().numpy()

        y_true_list.append(y)
        y_prob_list.append(prob)

        if count % 100 == 0:
            print(f"Processed {count} batches out of {length}")
        count +=  1

    Y_true = np.concatenate(y_true_list, axis=0).astype(int)
    Y_prob = np.concatenate(y_prob_list, axis=0).astype(float)
    return Y_true, Y_prob


def evaluate_multilabel(
    model: Union[SwinMIMICGraphClassifier, SwinMIMICClassifier],
    test_loader: torch.utils.data.DataLoader,
    device: Optional[str] = None,
    out_json: str = "results/test_eval.json",
    n_boot: int = 2000,
    seed: int = 42,
    n_bins: int = 15,
    save_npz: bool = False,
    out_npz: str = "results/preds_test.npz",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Y_true, Y_prob = collect_probs_and_labels(model, test_loader, device=device)
    Y_true_ablation, Y_prob_ablation = collect_probs_and_labels(model, test_loader, device=device, use_nudger=False)

    out_json = out_json.split('.')[0] + f"_{'_base_ft' if not isinstance(model, SwinMIMICGraphClassifier) else ''}" + ".json"
    out_npz = out_npz.split('.')[0] + f"_{'_base_ft' if not isinstance(model, SwinMIMICGraphClassifier) else ''}" + ".npz"

    results = bootstrap_multilabel(
        Y_true,
        Y_prob,
        label_names=MIMIC_LABELS,
        n_boot=n_boot,
        seed=seed,
        n_bins=n_bins,
        alpha=0.05,
    )

    results_ablation = bootstrap_multilabel(
        Y_true_ablation,
        Y_prob_ablation,
        label_names=MIMIC_LABELS,
        n_boot=n_boot,
        seed=seed,
        n_bins=n_bins,
        alpha=0.05,
    )

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    our_path_ablation = out_path.with_name(out_path.stem + "_ablation" + out_path.suffix)
    our_path_ablation.write_text(json.dumps(results_ablation, indent=2))

    if save_npz:
        npz_path = Path(out_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            Y_true=Y_true,
            Y_prob=Y_prob,
            label_names=np.array(MIMIC_LABELS, dtype=object),
        )

    return results
