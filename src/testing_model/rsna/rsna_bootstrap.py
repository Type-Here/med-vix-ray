import os
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss
)

from settings import MANUAL_GRAPH, MODELS_DIR
from src.med_vix_ray import SwinMIMICGraphClassifier
from src.testing_model.rsna.rsna_ft import adapt_model_to_rsna


# ----------------------------
# 1) Preprocess DICOM (RSNA)
# ----------------------------
def preprocess_rsna_dicom(dicom_path: str, image_size=(256, 256)) -> torch.Tensor:
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)

    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img = (img * 255).astype(np.uint8)

    img_pil = Image.fromarray(img).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.125), interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return transform(img_pil).unsqueeze(0)  # [1,C,H,W]


# ----------------------------
# 2) Dataset
# ----------------------------
class RSNADataset(torch.utils.data.Dataset):
    def __init__(self, dicom_dir: str, df_grouped: pd.DataFrame, image_size=(256, 256)):
        self.dicom_dir = dicom_dir
        self.df = df_grouped.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["patientId"]
        dcm_path = os.path.join(self.dicom_dir, f"{pid}.dcm")

        x = preprocess_rsna_dicom(dcm_path, self.image_size).squeeze(0)  # [C,H,W]
        # y: [Target, ternary]
        y = torch.tensor([row["Target"], row["ternary"]], dtype=torch.float32)
        return x, y, pid


# ----------------------------
# 3) Build grouped df (1 row per patientId)
# ----------------------------
def build_rsna_grouped_df(rsna_csv_binary: str, rsna_csv_tern: str) -> pd.DataFrame:
    df = pd.read_csv(rsna_csv_binary)  # patientId,x,y,width,height,Target
    df_tern = pd.read_csv(rsna_csv_tern)  # patientId,class

    df = df.merge(df_tern[["patientId", "class"]], on="patientId", how="left")

    df["ternary"] = df["class"].map({
        "Normal": 0,
        "No Lung Opacity / Not Normal": 1,
        "Lung Opacity": 2,
    }).fillna(1).astype(int)

    # 1 row per patientId
    grouped = (
        df.groupby("patientId", as_index=False)
          .agg(Target=("Target", "max"),
               ternary=("ternary", "first"))
    )
    grouped["Target"] = grouped["Target"].astype(int)
    grouped["ternary"] = grouped["ternary"].astype(int)
    return grouped


# ----------------------------
# 4) Load pretrained model
# ----------------------------
def load_pretrained_model() -> Tuple[SwinMIMICGraphClassifier, torch.device]:
    with open(MANUAL_GRAPH, "r") as f:
        graph_json = json.load(f)

    device = torch.device("cpu" if torch.version.hip else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SwinMIMICGraphClassifier(graph_json=graph_json, device=device).to(device)
    adapt_model_to_rsna(model, device=device)
    model.is_fine_tuning = False  # just to be safe

    save_dir = os.path.join(MODELS_DIR, "mvr-rsna")
    state_path = os.path.join(save_dir, "rsna_finetuned_model_state.pth")
    graph_path = os.path.join(save_dir, "rsna_finetuned_model_graph.json")

    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing model state: {state_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing graph json: {graph_path}")

    with open(graph_path, "r") as f:
        trained_graph = json.load(f)

    model.load_model_from_state(state_dict_path=state_path, graph_json=trained_graph)
    model.eval()
    return model, device


# ----------------------------
# 5) Metrics helpers
# ----------------------------
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    # y_true, y_pred: {0,1}; y_score in [0,1]
    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    # AUROC/AUPRC require both classes present, else crash
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
        out["brier"] = float(brier_score_loss(y_true, y_score))
        out["logloss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        out["brier"] = float("nan")
        out["logloss"] = float("nan")

    return out


def compute_ternary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # y_true,y_pred in {0,1,2}
    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return out


def percentile_ci(samples: np.ndarray, alpha=0.05) -> Dict[str, float]:
    lo = np.nanpercentile(samples, 100 * (alpha / 2))
    hi = np.nanpercentile(samples, 100 * (1 - alpha / 2))
    return {"lo": float(lo), "hi": float(hi)}


# ----------------------------
# 6) RSNA eval (point + bootstrap)
# ----------------------------
@torch.no_grad()
def rsna_inference_collect(model, loader, device) -> Dict[str, Any]:
    # Collect predictions
    y_true_bin, y_pred_bin, y_score_bin = [], [], []
    y_true_ter, y_pred_ter = [], []
    pids_all = []
    count = 0
    len_all = len(loader)
    for x, y, pids in loader:
        x = x.to(device)

        logits = model(x)                       # [B,14]
        probs = torch.sigmoid(logits).cpu().numpy()

        # RSNA pneumonia-like score: max of (Pneumonia idx 11, Lung Opacity idx 7, Consolidation idx 2)
        #score = np.maximum.reduce([probs[:, 11], probs[:, 7], probs[:, 2]])
        pred_bin = (probs > 0.5).astype(np.int32)

        # ternary mapping (discrete)
        # 0 = Normal (No Finding)
        # 2 = Lung Opacity / Pneumonia-like
        # 1 = Other
        #pred_ter = np.full((probs.shape[0],), 1, dtype=np.int32)
        #pred_ter[probs[:, 8] > 0.5] = 0  # No Finding
        #pred_ter[(probs[:, 11] > 0.5) | (probs[:, 7] > 0.5) | (probs[:, 2] > 0.5)] = 2

        y_np = y.cpu().numpy()
        yb = y_np[:, 0].astype(np.int32)
        #yt = y_np[:, 1].astype(np.int32)

        y_true_bin.append(yb)
        y_pred_bin.append(pred_bin)
        y_score_bin.append(probs)

        #y_true_ter.append(yt)
        #y_pred_ter.append(pred_ter)

        pids_all.extend(list(pids))

        count += 1
        if count % 100 == 0:
            print(f"[INFO] Processed {count}/{len_all} batches ({(count/len_all)*100:.1f}%)")

    return {
        "y_true_bin": np.concatenate(y_true_bin),
        "y_pred_bin": np.concatenate(y_pred_bin),
        "y_score_bin": np.concatenate(y_score_bin),
        #"y_true_ter": np.concatenate(y_true_ter),
        #"y_pred_ter": np.concatenate(y_pred_ter),
        "patientIds": pids_all
    }


def bootstrap_rsna_metrics(collected: Dict[str, Any], n_boot=200, alpha=0.05, seed=42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(collected["y_true_bin"])

    point_bin = compute_binary_metrics(collected["y_true_bin"], collected["y_pred_bin"], collected["y_score_bin"])
    #point_ter = compute_ternary_metrics(collected["y_true_ter"], collected["y_pred_ter"])

    # bootstrap distributions
    boot_bin = {k: [] for k in point_bin.keys()}
    #boot_ter = {k: [] for k in point_ter.keys()}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample rows
        ytb = collected["y_true_bin"][idx]
        ypb = collected["y_pred_bin"][idx]
        ysb = collected["y_score_bin"][idx]

        #ytt = collected["y_true_ter"][idx]
        #ypt = collected["y_pred_ter"][idx]

        mb = compute_binary_metrics(ytb, ypb, ysb)
        #mt = compute_ternary_metrics(ytt, ypt)

        for k, v in mb.items():
            boot_bin[k].append(v)
        #for k, v in mt.items():
        #    boot_ter[k].append(v)

    ci_bin = {k: percentile_ci(np.array(v), alpha=alpha) for k, v in boot_bin.items()}
    #ci_ter = {k: percentile_ci(np.array(v), alpha=alpha) for k, v in boot_ter.items()}

    out = {
        "point": {
            "binary": point_bin,
            #"ternary": point_ter,
            "n": int(n)
        },
        "ci": {
            "binary": ci_bin,
            #"ternary": ci_ter
        },
        "meta": {
            "n": int(n),
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "seed": int(seed),
            "bootstrap_unit": "rows",
            "ci_method": "percentile",
            "binary_score_definition": "max(sigmoid(logits[Pneumonia]), sigmoid(logits[LungOpacity]), "
                                       "sigmoid(logits[Consolidation]))",
            #"ternary_mapping": {
            #    "0": "No Finding (if P(NoFinding)>0.5)",
            #    "2": "Pneumonia-like (if any of Pneumonia/LungOpacity/Consolidation >0.5)",
            #    "1": "Other"
            #}
        }
    }
    return out


# ----------------------------
# 7) Main
# ----------------------------
if __name__ == "__main__":
    rsna_dir = os.getenv("RSNA_DIR", None)
    rsna_csv_binary = os.getenv("RSNA_CSV_PATH_BINARY", None)
    rsna_csv_tern = os.getenv("RSNA_CSV_PATH_TERN", None)

    if not rsna_dir or not rsna_csv_binary or not rsna_csv_tern:
        raise RuntimeError("Set env vars: RSNA_DIR, RSNA_CSV_PATH_BINARY, RSNA_CSV_PATH_TERN")

    model, device = load_pretrained_model()

    df_grouped = build_rsna_grouped_df(rsna_csv_binary, rsna_csv_tern)

    # Fixed Subset for faster testing (5337 unique patientIds in total)
    # df_grouped = df_grouped.sample(n=5337, random_state=42).reset_index(drop=True)

    dataset = RSNADataset(rsna_dir, df_grouped, image_size=(256, 256))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[INFO] RSNA rows (unique patientId): {len(dataset)}")

    collected = rsna_inference_collect(model, loader, device)

    # save preds
    np.savez_compressed(
        "../../test_results/rsna/rsna_preds.npz",
        y_true_bin=collected["y_true_bin"],
        y_pred_bin=collected["y_pred_bin"],
        y_score_bin=collected["y_score_bin"],
        #y_true_ter=collected["y_true_ter"],
        #y_pred_ter=collected["y_pred_ter"],
        patientIds=np.array(collected["patientIds"], dtype=object),
    )

    boot = bootstrap_rsna_metrics(collected, n_boot=200, alpha=0.05, seed=42)

    with open("../../test_results/rsna/rsna_bootstrap.json", "w") as f:
        json.dump(boot, f, indent=2)

    with open("../../test_results/rsna/rsna_point.json", "w") as f:
        json.dump(boot["point"], f, indent=2)

    print("[DONE] Saved: rsna_point.json, rsna_bootstrap.json, rsna_preds.npz")
    print("Binary (point):", boot["point"]["binary"])
    #print("Ternary (point):", boot["point"]["ternary"])
