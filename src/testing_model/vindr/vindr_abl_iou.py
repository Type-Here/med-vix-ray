import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)

from settings import MANUAL_GRAPH, MODELS_DIR
from src.med_vix_ray import SwinMIMICGraphClassifier


# ----------------------------
# 0) Label mapping (VinDr -> MIMIC 14)
# ----------------------------
VINDR_TO_MIMIC = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Enlarged PA": "Enlarged Cardiomediastinum",
    "Mediastinal shift": "Enlarged Cardiomediastinum",
    "Clavicle fracture": "Fracture",
    "Rib fracture": "Fracture",
    "Lung cavity": "Lung Lesion",
    "Lung cyst": "Lung Lesion",
    "Nodule/Mass": "Lung Lesion",
    "Lung tumor": "Lung Lesion",
    "Other lesion": "Lung Lesion",
    "Lung Opacity": "Lung Opacity",
    "Infiltration": "Lung Opacity",
    "Pleural effusion": "Pleural Effusion",
    "Pleural thickening": "Pleural Other",
    "Pneumothorax": "Pneumothorax",
    "Pneumonia": "Pneumonia",
    "Tuberculosis": "Lung Lesion",
    "No finding": "No Finding",
}

MIMIC_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices"
]


# ----------------------------
# 1) VinDr preprocessing (same as your inference)
# ----------------------------
def preprocess_vindr_dicom(dicom_path: str, image_size=(256, 256)) -> torch.Tensor:
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
# 2) Resize/crop box transform: original coords -> 256x256 coords
#     Must match: Resize to S=round(1.125*256)=288 then CenterCrop 256
# ----------------------------
def transform_box_to_256(
    box_xyxy: List[float],
    orig_w: int,
    orig_h: int,
    resize_to: int = 288,
    crop_to: int = 256
) -> Optional[List[float]]:
    """
    box_xyxy: [x_min, y_min, x_max, y_max] in original pixel coords
    returns transformed box in 256x256 coords (clipped), or None if invalid
    """

    x1, y1, x2, y2 = box_xyxy

    # scale to resize_to
    sx = resize_to / float(orig_w)
    sy = resize_to / float(orig_h)

    x1 *= sx; x2 *= sx
    y1 *= sy; y2 *= sy

    # center crop crop_to from resize_to
    dx = (resize_to - crop_to) / 2.0
    dy = (resize_to - crop_to) / 2.0

    x1 -= dx; x2 -= dx
    y1 -= dy; y2 -= dy

    # clip to [0, crop_to]
    x1 = max(0.0, min(float(crop_to - 1), x1))
    y1 = max(0.0, min(float(crop_to - 1), y1))
    x2 = max(0.0, min(float(crop_to - 1), x2))
    y2 = max(0.0, min(float(crop_to - 1), y2))

    # validate
    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom)


def attn_to_box(attn: np.ndarray, thr: float = 0.9, use_percentile: bool = True) -> List[float]:
    """
    attn: [H,W] in [0,1]
    returns [x1,y1,x2,y2] in 0..255
    """
    if use_percentile:
        t = np.quantile(attn, thr)  # thr=0.9 -> 90th percentile
    else:
        t = thr

    mask = attn > t
    coords = np.argwhere(mask)
    if coords.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


# ----------------------------
# 3) Dataset (also returns orig shape for bbox transform)
# ----------------------------
class VinDrDataset(Dataset):
    def __init__(self, dicom_dir: str, df_labels: pd.DataFrame, image_size=(256, 256)):
        self.dicom_dir = dicom_dir
        self.df_labels = df_labels
        self.image_ids = list(df_labels.index.values)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        dicom_path = os.path.join(self.dicom_dir, f"{image_id}.dicom")

        dcm = pydicom.dcmread(dicom_path)
        orig_h, orig_w = dcm.pixel_array.shape[:2]

        x = preprocess_vindr_dicom(dicom_path, self.image_size).squeeze(0)
        y = torch.tensor(self.df_labels.loc[image_id].values.astype(np.float32))
        return x, y, image_id, (orig_w, orig_h)


# ----------------------------
# 4) Build MIMIC-14 label matrix from VinDr label CSV
# ----------------------------
def build_vindr_mimic14_labels(vindr_df: pd.DataFrame) -> pd.DataFrame:
    """
    vindr_df: rows per image (can be multiple rows); columns are VinDr labels 0/1
    returns df_out indexed by image_id with 14 columns in MIMIC_LABELS
    """
    vindr_label_cols = [c for c in VINDR_TO_MIMIC.keys() if c in vindr_df.columns]
    if len(vindr_label_cols) == 0:
        raise ValueError("No VinDr label columns found in the provided CSV (check column names).")

    # 1 row per image, OR over rows
    g = vindr_df.groupby("image_id")[vindr_label_cols].max()

    df_out = pd.DataFrame(0, index=g.index, columns=MIMIC_LABELS, dtype=np.int32)

    # map each vindr column into mimic column via OR
    for vindr_col in vindr_label_cols:
        mimic_label = VINDR_TO_MIMIC[vindr_col]
        df_out[mimic_label] = np.maximum(df_out[mimic_label].values, g[vindr_col].values.astype(np.int32))

    return df_out


# ----------------------------
# 5) Load pretrained model
# ----------------------------
def load_pretrained_model() -> Tuple[SwinMIMICGraphClassifier, torch.device]:
    with open(MANUAL_GRAPH, "r") as f:
        graph_json = json.load(f)

    device = torch.device("cpu" if torch.version.hip else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SwinMIMICGraphClassifier(graph_json=graph_json, device=device).to(device)

    save_dir = os.path.join(MODELS_DIR, "med-vix-ray")
    state_path = os.path.join(save_dir, "med_vixray_model_state.pth")
    graph_path = os.path.join(save_dir, "med_vixray_model_graph.json")

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
# 6) Multilabel metrics (point + bootstrap)
# ----------------------------
def multilabel_point_metrics(y_true: np.ndarray, y_score: np.ndarray, thr=0.5) -> Dict[str, Any]:
    y_pred = (y_score > thr).astype(np.int32)

    out = {
        "n": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # AUROC/AUPRC per label (skip labels with single class)
    aurocs = []
    auprcs = []
    per_label = {"auroc": {}, "auprc": {}}

    for j, name in enumerate(MIMIC_LABELS):
        yt = y_true[:, j]
        ys = y_score[:, j]
        if len(np.unique(yt)) < 2:
            per_label["auroc"][name] = float("nan")
            per_label["auprc"][name] = float("nan")
            continue
        a = roc_auc_score(yt, ys)
        p = average_precision_score(yt, ys)
        per_label["auroc"][name] = float(a)
        per_label["auprc"][name] = float(p)
        aurocs.append(a)
        auprcs.append(p)

    out["auroc_macro"] = float(np.nanmean(aurocs)) if len(aurocs) else float("nan")
    out["auprc_macro"] = float(np.nanmean(auprcs)) if len(auprcs) else float("nan")
    out["per_label"] = per_label
    return out


def percentile_ci(samples: np.ndarray, alpha=0.05) -> Dict[str, float]:
    lo = np.nanpercentile(samples, 100 * (alpha / 2))
    hi = np.nanpercentile(samples, 100 * (1 - alpha / 2))
    return {"lo": float(lo), "hi": float(hi)}


def multilabel_bootstrap(y_true: np.ndarray, y_score: np.ndarray, n_boot=200, alpha=0.05, seed=42, thr=0.5) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]

    point = multilabel_point_metrics(y_true, y_score, thr=thr)

    # scalar keys
    scalar_keys = ["exact_match", "f1_macro", "f1_micro", "precision_macro", "recall_macro", "auroc_macro", "auprc_macro"]
    boot_scalars = {k: [] for k in scalar_keys}

    # per label keys
    boot_auroc = {name: [] for name in MIMIC_LABELS}
    boot_auprc = {name: [] for name in MIMIC_LABELS}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = multilabel_point_metrics(y_true[idx], y_score[idx], thr=thr)

        for k in scalar_keys:
            boot_scalars[k].append(m.get(k, np.nan))

        for name in MIMIC_LABELS:
            boot_auroc[name].append(m["per_label"]["auroc"][name])
            boot_auprc[name].append(m["per_label"]["auprc"][name])

    ci = {
        "scalars": {k: percentile_ci(np.array(v), alpha=alpha) for k, v in boot_scalars.items()},
        "per_label": {
            "auroc": {name: percentile_ci(np.array(v), alpha=alpha) for name, v in boot_auroc.items()},
            "auprc": {name: percentile_ci(np.array(v), alpha=alpha) for name, v in boot_auprc.items()},
        }
    }

    return {
        "point": point,
        "ci": ci,
        "meta": {
            "n": int(n),
            "n_labels": int(y_true.shape[1]),
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "seed": int(seed),
            "bootstrap_unit": "rows",
            "ci_method": "percentile",
            "threshold": float(thr),
        }
    }


# ----------------------------
# 7) IoU evaluation using VinDr boxes
# ----------------------------
def load_boxes(boxes_csv: str) -> pd.DataFrame:
    """
    expects columns: image_id,class_name,x_min,y_min,x_max,y_max
    """
    df = pd.read_csv(boxes_csv)
    needed = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Boxes CSV missing columns. Need: {needed}")
    return df


def compute_iou_stats(
    attn_maps: Dict[str, np.ndarray],  # image_id -> [256,256] attention
    orig_shapes: Dict[str, Tuple[int,int]],  # image_id -> (orig_w, orig_h)
    boxes_df: pd.DataFrame,
    target_classes: List[str],
    thr_q: float = 0.9,
    use_percentile: bool = True
) -> Dict[str, Any]:
    """
    For each image with at least one GT box in target_classes:
      pred_box from attention map
      gt_boxes transformed to 256
      iou = max IoU over GT boxes
    Returns per-image list of IoU and recall@ thresholds.
    """
    boxes_df = boxes_df[boxes_df["class_name"].isin(target_classes)].copy()
    grouped = boxes_df.groupby("image_id")

    ious = []
    valid_ids = []

    for image_id, g in grouped:
        if image_id not in attn_maps or image_id not in orig_shapes:
            continue

        attn = attn_maps[image_id]
        pred_box = attn_to_box(attn, thr=thr_q, use_percentile=use_percentile)

        ow, oh = orig_shapes[image_id]

        gt_boxes = []
        for _, r in g.iterrows():
            tb = transform_box_to_256(
                [float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])],
                orig_w=ow, orig_h=oh
            )
            if tb is not None:
                gt_boxes.append(tb)

        if len(gt_boxes) == 0:
            continue

        # if pred_box empty -> IoU=0
        if pred_box[2] <= pred_box[0] or pred_box[3] <= pred_box[1]:
            best = 0.0
        else:
            best = max(iou_xyxy(pred_box, b) for b in gt_boxes)

        ious.append(best)
        valid_ids.append(image_id)

    ious = np.array(ious, dtype=np.float32)

    def recall_at(t: float) -> float:
        if len(ious) == 0:
            return float("nan")
        return float((ious >= t).mean())

    return {
        "n_images_with_boxes": int(len(ious)),
        "mean_iou": float(np.mean(ious)) if len(ious) else float("nan"),
        "median_iou": float(np.median(ious)) if len(ious) else float("nan"),
        "recall_iou_ge_0.10": recall_at(0.10),
        "recall_iou_ge_0.25": recall_at(0.25),
        "recall_iou_ge_0.50": recall_at(0.50),
        "ious": ious,
        "image_ids": valid_ids
    }


def bootstrap_iou(iou_vals: np.ndarray, n_boot=200, alpha=0.05, seed=42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(iou_vals)
    if n == 0:
        return {"point": {}, "ci": {}, "meta": {"n": 0}}

    mean_samples = []
    med_samples = []
    r10_samples = []
    r25_samples = []
    r50_samples = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = iou_vals[idx]
        mean_samples.append(np.mean(s))
        med_samples.append(np.median(s))
        r10_samples.append((s >= 0.10).mean())
        r25_samples.append((s >= 0.25).mean())
        r50_samples.append((s >= 0.50).mean())

    point = {
        "mean_iou": float(np.mean(iou_vals)),
        "median_iou": float(np.median(iou_vals)),
        "recall_iou_ge_0.10": float((iou_vals >= 0.10).mean()),
        "recall_iou_ge_0.25": float((iou_vals >= 0.25).mean()),
        "recall_iou_ge_0.50": float((iou_vals >= 0.50).mean()),
        "n": int(n)
    }

    ci = {
        "mean_iou": percentile_ci(np.array(mean_samples), alpha=alpha),
        "median_iou": percentile_ci(np.array(med_samples), alpha=alpha),
        "recall_iou_ge_0.10": percentile_ci(np.array(r10_samples), alpha=alpha),
        "recall_iou_ge_0.25": percentile_ci(np.array(r25_samples), alpha=alpha),
        "recall_iou_ge_0.50": percentile_ci(np.array(r50_samples), alpha=alpha),
    }

    return {
        "point": point,
        "ci": ci,
        "meta": {"n": int(n), "n_boot": int(n_boot), "alpha": float(alpha), "seed": int(seed)}
    }


# ----------------------------
# 8) Main
# ----------------------------
@torch.no_grad()
def run_vindr(model, device, loader) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray], Dict[str, Tuple[int,int]]]:
    all_scores = []
    all_true = []
    all_ids = []

    attn_maps = {}     # image_id -> [256,256]
    orig_shapes = {}   # image_id -> (orig_w, orig_h)

    model.eval()
    for x, y, image_ids, shapes in loader:
        x = x.to(device)
        logits = model(x)  # [B,14]
        probs = torch.sigmoid(logits).cpu().numpy()

        all_scores.append(probs)
        all_true.append(y.numpy())
        all_ids.extend(list(image_ids))

        # attention maps from model (assumes model.att_maps_batch set by forward)
        # your code suggests it's [B,H,W] or [B,1,H,W] depending on implementation
        am = getattr(model, "att_maps_batch", None)
        if am is not None:
            am = am.detach().cpu().numpy()
            # normalize shape to [B,256,256]
            if am.ndim == 4:
                am = am[:, 0]
            for i, iid in enumerate(image_ids):
                attn_maps[iid] = am[i].astype(np.float32)

        for iid, (ow, oh) in zip(image_ids, shapes):
            orig_shapes[iid] = (int(ow), int(oh))

    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    return y_true, y_score, all_ids, attn_maps, orig_shapes


if __name__ == "__main__":
    vindr_dir = os.getenv("VINDR_DIR", None)
    vindr_labels_csv = os.getenv("VINDR_LABELS_CSV", None)
    vindr_boxes_csv = os.getenv("VINDR_BOXES_CSV", None)

    if not vindr_dir or not vindr_labels_csv:
        raise RuntimeError("Set env vars: VINDR_DIR and VINDR_LABELS_CSV. Optionally VINDR_BOXES_CSV for IoU.")

    model, device = load_pretrained_model()

    vindr_df = pd.read_csv(vindr_labels_csv)
    if "image_id" not in vindr_df.columns:
        raise ValueError("VinDr labels CSV must contain column 'image_id'.")

    df_out = build_vindr_mimic14_labels(vindr_df)
    df_out.to_csv("vindr_mimic14_labels.csv")
    print(f"[INFO] Built MIMIC-14 labels: {df_out.shape} saved to vindr_mimic14_labels.csv")

    dataset = VinDrDataset(vindr_dir, df_out, image_size=(256, 256))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[INFO] VinDr images: {len(dataset)}")

    y_true, y_score, image_ids, attn_maps, orig_shapes = run_vindr(model, device, loader)

    # Multilabel metrics + bootstrap
    metrics_boot = multilabel_bootstrap(y_true, y_score, n_boot=200, alpha=0.05, seed=42, thr=0.5)

    with open("vindr_multilabel_bootstrap.json", "w") as f:
        json.dump(metrics_boot, f, indent=2)
    print("[DONE] Saved vindr_multilabel_bootstrap.json")

    # Optional IoU
    if vindr_boxes_csv and os.path.exists(vindr_boxes_csv):
        boxes_df = load_boxes(vindr_boxes_csv)

        # You can extend this list, but start with Pneumothorax because it's typically boxed.
        target_classes = ["Pneumothorax"]

        iou_stats = compute_iou_stats(
            attn_maps=attn_maps,
            orig_shapes=orig_shapes,
            boxes_df=boxes_df,
            target_classes=target_classes,
            thr_q=0.90,              # 90th percentile mask
            use_percentile=True
        )

        iou_boot = bootstrap_iou(iou_stats["ious"], n_boot=200, alpha=0.05, seed=42)

        out_iou = {
            "target_classes": target_classes,
            "threshold_percentile": 0.90,
            "point": {k: v for k, v in iou_stats.items() if k not in {"ious", "image_ids"}},
            "bootstrap": iou_boot
        }

        with open("vindr_iou_bootstrap.json", "w") as f:
            json.dump(out_iou, f, indent=2)
        print("[DONE] Saved vindr_iou_bootstrap.json")
    else:
        print("[INFO] No VINDR_BOXES_CSV provided; skipping IoU.")
