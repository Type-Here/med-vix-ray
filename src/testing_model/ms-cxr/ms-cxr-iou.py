import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.testing_model.vindr.vindr_abl_iou import (iou_xyxy, dice_xyxy, attn_to_box, attn_to_box_dilated,
                   compute_center_point_recall, bootstrap_iou)
from src.testing_model.vindr.vindr_abl_iou import load_pretrained_model

# ----------------------------
# MS-CXR supported classes (8)
# ----------------------------
MS_CXR_8 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


# ----------------------------
# Correct bbox transform for your MIMIC preprocessing:
# Resize(int(1.125*256)) with aspect ratio preserved + CenterCrop(256).
# IMPORTANT: torchvision.Resize(int) uses "short side -> size".
# ----------------------------
def transform_box_to_256_mimic(
    box_xyxy: List[float],
    orig_w: int,
    orig_h: int,
    resize_short: int = 288,
    crop_to: int = 256,
    view_position: str = "AP",
) -> Optional[List[float]]:
    x1, y1, x2, y2 = map(float, box_xyxy)

    # Resize(short_side=resize_short), preserve aspect ratio
    scale = resize_short / float(min(orig_w, orig_h))
    new_w = float(orig_w) * scale
    new_h = float(orig_h) * scale

    x1 *= scale
    x2 *= scale
    y1 *= scale
    y2 *= scale

    # CenterCrop(crop_to)
    dx = (new_w - crop_to) / 2.0
    dy = (new_h - crop_to) / 2.0
    x1 -= dx
    x2 -= dx
    y1 -= dy
    y2 -= dy

    # Clip
    x1 = max(0.0, min(crop_to - 1.0, x1))
    y1 = max(0.0, min(crop_to - 1.0, y1))
    x2 = max(0.0, min(crop_to - 1.0, x2))
    y2 = max(0.0, min(crop_to - 1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # For MS-CXR, unless loading per-image view_position from MIMIC metadata,
    # keep it fixed to AP to avoid mismatch.
    if view_position == "PA":
        W = crop_to - 1.0
        fx1 = W - x2
        fx2 = W - x1
        x1, x2 = fx1, fx2

    return [x1, y1, x2, y2]


# ----------------------------
# Load MS-CXR official CSV (generated with their converter)
# and convert xywh -> xyxy.
# ----------------------------
def load_mscxr_csv(ms_csv_path: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(ms_csv_path, sep=None, engine="python")  # handles tabs/commas

    required = {
        "dicom_id", "category_name", "path",
        "x", "y", "w", "h",
        "image_width", "image_height",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MS-CXR CSV missing columns: {missing}")

    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()].copy()

    df = df.copy()
    df["image_id"] = df["dicom_id"].astype(str)
    df["class_name"] = df["category_name"].astype(str)

    # Keep only supported 8 classes
    df = df[df["class_name"].isin(MS_CXR_8)].copy()

    # xywh -> xyxy
    df["x_min"] = df["x"].astype(float)
    df["y_min"] = df["y"].astype(float)
    df["x_max"] = (df["x"] + df["w"]).astype(float)
    df["y_max"] = (df["y"] + df["h"]).astype(float)

    df["orig_w"] = df["image_width"].astype(int)
    df["orig_h"] = df["image_height"].astype(int)

    return df.reset_index(drop=True)


# ----------------------------
# Dataset for MS-CXR (images referenced by CSV 'path')
# ----------------------------
class MSCXRDataset(Dataset):
    def __init__(
        self,
        mimic_root: str,
        boxes_df: pd.DataFrame,
        preprocess_fn,
        image_size=(256, 256),
        channels_mode="RGB",
        view_position="AP",
    ):
        self.mimic_root = mimic_root
        self.preprocess_fn = preprocess_fn
        self.image_size = image_size
        self.channels_mode = channels_mode
        self.view_position = view_position

        # one row per image (keep path + original sizes)
        self.images = (
            boxes_df[["image_id", "path", "orig_w", "orig_h"]]
            .drop_duplicates("image_id")
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        r = self.images.iloc[idx]
        image_id = str(r["image_id"])
        rel_path = str(r["path"]).lstrip("/")  # just in case
        img_path = os.path.join(self.mimic_root, rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path)

        x = self.preprocess_fn(
            img,
            channels_mode=self.channels_mode,
            image_size=self.image_size,
            view_position=r["view_position"] if "view_position" in r else self.view_position,
            augment=False,
            is_train=False,
        )

        ow = int(r["orig_w"])
        oh = int(r["orig_h"])
        return x, image_id, (ow, oh)


# ----------------------------
# Inference: produce y_score (optional) and attention maps
# ----------------------------
@torch.no_grad()
def run_mscxr(
    model,
    device,
    loader,
    n_labels: int = 14,   # your model outputs 14
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
    model.eval()

    all_scores = []
    all_ids: List[str] = []
    attn_maps: Dict[str, np.ndarray] = {}
    orig_shapes: Dict[str, Tuple[int, int]] = {}

    for x, image_ids, shapes in loader:
        x = x.to(device)
        logits = model(x)                   # [B,14]
        probs = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(probs)
        all_ids.extend([str(i) for i in image_ids])

        am = getattr(model, "att_maps_batch", None)
        if am is None:
            raise RuntimeError("Model does not expose att_maps_batch; cannot compute IoU from attention maps.")
        am = am.detach().cpu().numpy()
        if am.ndim == 4:
            am = am[:, 0]  # [B,256,256]

        for idx, iid in enumerate(image_ids):
            iid = str(iid)
            attn_maps[iid] = am[idx].astype(np.float32)
            # shapes may be:
            #  - a sequence of per-sample (w,h) entries (list/tuple or tensor [B,2])
            orig_shapes[iid] = (int(shapes[0][idx]), int(shapes[1][idx]))

    y_score = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0, n_labels), dtype=np.float32)
    return y_score, all_ids, attn_maps, orig_shapes


# ----------------------------
# IoU stats for MS-CXR using correct transform
# ----------------------------
def compute_iou_stats_mscxr(
    attn_maps: Dict[str, np.ndarray],
    orig_shapes: Dict[str, Tuple[int, int]],
    boxes_df: pd.DataFrame,
    target_classes: List[str],
    thr_q: float,
    use_percentile: bool,
    attn_to_box_fn,
    measure_fn,
    view_position: str = "AP",
) -> Dict[str, Any]:
    df = boxes_df[boxes_df["class_name"].isin(target_classes)].copy()
    grouped = df.groupby("image_id")

    ious = []
    valid_ids = []

    for image_id, g in grouped:
        image_id = str(image_id)
        if image_id not in attn_maps or image_id not in orig_shapes:
            continue

        attn = attn_maps[image_id]
        pred_box = attn_to_box_fn(attn, thr=thr_q, use_percentile=use_percentile)

        ow, oh = orig_shapes[image_id]

        gt_boxes = []
        for _, r in g.iterrows():
            tb = transform_box_to_256_mimic(
                [r["x_min"], r["y_min"], r["x_max"], r["y_max"]],
                orig_w=int(ow),
                orig_h=int(oh),
                resize_short=288,
                crop_to=256,
                view_position=r["view_position"] if "view_position" in r else view_position,
            )
            if tb is not None:
                gt_boxes.append(tb)

        if not gt_boxes:
            continue

        if pred_box[2] <= pred_box[0] or pred_box[3] <= pred_box[1]:
            best = 0.0
        else:
            best = max(measure_fn(pred_box, b) for b in gt_boxes)

        ious.append(best)
        valid_ids.append(image_id)

    ious = np.array(ious, dtype=np.float32)

    def recall_at(t: float) -> float:
        return float((ious >= t).mean()) if len(ious) else float("nan")

    return {
        "n_images_with_boxes": int(len(ious)),
        "mean_iou": float(np.mean(ious)) if len(ious) else float("nan"),
        "median_iou": float(np.median(ious)) if len(ious) else float("nan"),
        "recall_iou_ge_0.10": recall_at(0.10),
        "recall_iou_ge_0.25": recall_at(0.25),
        "recall_iou_ge_0.50": recall_at(0.50),
        "ious": ious,
        "image_ids": valid_ids,
    }


# ----------------------------
# Optional: disk-backed dump like your VinDr lazy loader
# ----------------------------
def save_mscxr_json(raw_out: Dict[str, Any], out_json_path: str) -> None:
    with open(out_json_path, "w") as f:
        json.dump(raw_out, f, indent=2)
    print(f"[DONE] Saved raw outputs to {out_json_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    """
    Env vars:
      - MIMIC_JPG_ROOT: directory that contains "files/..." (NOT the 'files' folder itself)
        Example: /data/mimic-cxr-jpg
      - MS_CXR_CSV_PATH: path to MS_CXR_Local_Alignment_v1.0.0.csv (official converter output)
      - MS_CXR_SPLIT: optional {train|valid|test} if column exists
      - MS_CXR_RAW_JSON: optional path to save big raw json (like vindr_raw_outputs)
      - MS_CXR_RESULTS_JSON: output metrics json
    """
    mimic_root = os.getenv("MIMIC_DATASET_PATH")
    if mimic_root is None:
        raise RuntimeError("Set env var MIMIC_DATASET_PATH to the "
                           "root directory containing 'files/' for MIMIC-CXR.")
    # remove redundant "files/" if end of path
    if mimic_root and mimic_root.endswith("files"):
        mimic_root = mimic_root[:-len("files")].rstrip("/")

    ms_csv = os.getenv("MS_CXR_CSV_PATH")
    split = "test"  # default to test split
    if "MS_CXR_SPLIT" in os.environ:
        split = os.getenv("MS_CXR_SPLIT").lower()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid MS_CXR_SPLIT: {split}. Must be one of train/valid/test.")

    raw_json = os.getenv("MS_CXR_RAW_JSON", None)
    results_json = os.getenv("MS_CXR_RESULTS_JSON", "mscxr_iou_results.json")

    if not mimic_root or not ms_csv:
        raise RuntimeError("Set env vars: MIMIC_DATASET_PATH and MS_CXR_CSV_PATH")

    # ---- Import your existing code (no rewrites) ----

    # Change this import to wherever your preprocess_image lives
    from src.preprocess import preprocess_image

    # Load MS-CXR boxes
    boxes_df = load_mscxr_csv(ms_csv, split=split)
    print(f"[INFO] MS-CXR rows={len(boxes_df)} images={boxes_df['image_id'].nunique()} split={split}")

    # Load model
    model, device = load_pretrained_model()

    # Load test metaadata
    import dataset.dataset_handle as dh
    test_metadata = dh.fetch_metadata(phase="test", full_data=True, verify_existence=False)

    # Retrieve view_position for each image_id from test_metadata (if available)
    # and add a column to boxes_df. If not available, default to "AP".
    if "view_position" in test_metadata[0]:
        id_to_view = {str(m["dicom_id"]): m["view_position"] for m in test_metadata}
        boxes_df["view_position"] = boxes_df["image_id"].astype(str).map(id_to_view).fillna("AP")
        print(f"[INFO] Added view_position column to boxes_df based on test metadata.")
    else:
        boxes_df["view_position"] = "AP"
        print(f"[INFO] view_position not found in test metadata; defaulting to AP for all images.")

    # Dataset/Loader
    ds = MSCXRDataset(
        mimic_root=mimic_root,
        boxes_df=boxes_df,
        preprocess_fn=preprocess_image,
        image_size=(256, 256),
        channels_mode="RGB",
        view_position="AP",   # keep fixed unless you also load per-image view_position
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[INFO] Running MS-CXR inference on {len(ds)} images...")

    # Run inference
    y_score, image_ids, attn_maps, orig_shapes = run_mscxr(model, device, loader, n_labels=14)
    print(f"[INFO] Done. attn_maps={len(attn_maps)} orig_shapes={len(orig_shapes)} y_score={y_score.shape}")

    # Optionally save a big json for reproducibility / lazy conversion
    if raw_json:
        raw_out = {
            "y_score": y_score.tolist(),
            "image_ids": image_ids,
            "attn_maps": {iid: attn_maps[iid].tolist() for iid in attn_maps},
            "orig_shapes": {iid: orig_shapes[iid] for iid in orig_shapes},
            "meta": {
                "dataset": "MS-CXR",
                "split": split,
                "mimic_root": mimic_root,
                "ms_csv": ms_csv,
            }
        }
        save_mscxr_json(raw_out, raw_json)

    # Localization metrics
    target_classes = MS_CXR_8
    thr = 0.95

    print("\n=== LOCALIZATION METRICS (MS-CXR) ===\n")

    # 1) Pointing game / center-point recall (your existing fn)
    pt = compute_center_point_recall(attn_maps, orig_shapes, boxes_df, target_classes)
    print(f"Center Point Recall: {pt['point_recall']:.3f} ({pt['n_images']} images)")

    # 2) IoU tight
    iou_tight = compute_iou_stats_mscxr(
        attn_maps, orig_shapes, boxes_df, target_classes,
        thr_q=thr, use_percentile=True,
        attn_to_box_fn=attn_to_box,
        measure_fn=iou_xyxy,
        view_position="AP",
    )
    print(f"IoU tight: mean={iou_tight['mean_iou']:.4f}, median={iou_tight['median_iou']:.4f}, "
          f"R@0.10={iou_tight['recall_iou_ge_0.10']:.3f}")

    # 3) IoU dilated
    iou_dil = compute_iou_stats_mscxr(
        attn_maps, orig_shapes, boxes_df, target_classes,
        thr_q=thr, use_percentile=True,
        attn_to_box_fn=attn_to_box_dilated,
        measure_fn=iou_xyxy,
        view_position="AP",
    )
    print(f"IoU dilated: mean={iou_dil['mean_iou']:.4f}, median={iou_dil['median_iou']:.4f}")

    # 4) Dice dilated
    dice = compute_iou_stats_mscxr(
        attn_maps, orig_shapes, boxes_df, target_classes,
        thr_q=thr, use_percentile=True,
        attn_to_box_fn=attn_to_box_dilated,
        measure_fn=dice_xyxy,
        view_position="AP",
    )
    print(f"Dice dilated: mean={dice['mean_iou']:.4f}, median={dice['median_iou']:.4f}")

    # Bootstrap on tight IoU
    boot = bootstrap_iou(iou_tight["ious"], n_boot=200, alpha=0.05, seed=42)

    out = {
        "dataset": "MS-CXR",
        "split": split,
        "target_classes": target_classes,
        "threshold_quantile": thr,
        "point_recall": pt,
        "iou_tight": {k: v for k, v in iou_tight.items() if k not in {"ious", "image_ids"}},
        "iou_dilated": {k: v for k, v in iou_dil.items() if k not in {"ious", "image_ids"}},
        "dice_dilated": {k: v for k, v in dice.items() if k not in {"ious", "image_ids"}},
        "bootstrap_iou_tight": boot,
        "meta_notes": [
            "BBoxes transformed using Resize(short=288, aspect-preserving) + CenterCrop(256) to match preprocessing.",
            "PA flipping disabled (view_position fixed to AP) unless per-image metadata is provided.",
            "Multiple GT bboxes per image/category are handled using max overlap (best-match).",
        ],
    }

    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        return obj

    with open(results_json, "w") as f:
        json.dump(_make_serializable(out), f, indent=2)

    print(f"\n[DONE] Saved metrics to {results_json}\n")


if __name__ == "__main__":
    main()