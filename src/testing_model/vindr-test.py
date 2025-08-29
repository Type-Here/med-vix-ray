import json
import os

import pydicom
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, \
    average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from settings import MANUAL_GRAPH, MODELS_DIR
from src.med_vix_ray import SwinMIMICGraphClassifier


class VinDrDataset(torch.utils.data.Dataset):
    def __init__(self, dicom_paths, df_out, image_size=(256, 256)):
        self.dicom_paths = dicom_paths
        self.df_out = df_out
        self.image_size = image_size

    def __len__(self):
        return len(self.dicom_paths)

    def __getitem__(self, idx):
        dicom_path, image_id = self.dicom_paths[idx]
        x = preprocess_vindr_dicom(dicom_path, self.image_size)
        y = torch.tensor(self.df_out.loc[image_id].values.astype(np.float32))
        return x.squeeze(0), y, image_id



def preprocess_vindr_dicom(dicom_path, image_size=(256, 256)):
    """
    Preprocess a VinDr-CXR DICOM image for inference.
    Args:
        dicom_path (str): path to DICOM file.
        image_size (tuple): target size for the model.
    Returns:
        torch.Tensor: processed image tensor [1,C,H,W]
    """
    # Load DICOM
    dcm = pydicom.dcmread(dicom_path)

    # Extract pixel array
    img = dcm.pixel_array.astype(np.float32)

    # Windowing (optional, for consistent visualization)
    # Basic min-max normalization
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img = (img * 255).astype(np.uint8)

    # Convert to PIL Image
    img_pil = Image.fromarray(img).convert("RGB")  # use RGB because your model is trained on 3 channels

    # Define VinDr-specific transforms (similar to your training pipeline)
    transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.125), interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(img_pil).unsqueeze(0)  # add batch dimension [1, C, H, W]
    return tensor


# Define mapping: VINDR label → MIMIC label index
vindr_to_mimic_mapping = {
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
    "Pulmonary fibrosis": "Pleural Other",
    "ILD": "Pleural Other",
    "Pneumothorax": "Pneumothorax",
    "Pneumonia": "Pneumonia",
    "Tuberculosis": "Lung Lesion",  # Opzionale
    "No finding": "No Finding",
    # Labels without mapping are ignored
}

# MIMIC labels in order (needed for index lookup)
MIMIC_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
                "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
                "Pneumonia", "Pneumothorax", "Support Devices"]  # Support Devices → no match in VINDR

# Function to map VINDR row to MIMIC format row
def map_vindr_row_to_mimic(vindr_row: dict) -> list:
    """
    Given a VINDR row (label: 0/1), returns a MIMIC row (list of 0/1 values in correct order).
    If multiple VINDR labels map to one MIMIC label, we take OR across them.
    """
    mimic_row = [0] * len(MIMIC_LABELS)
    for vindr_label, value in vindr_row.items():
        if vindr_label in vindr_to_mimic_mapping:
            mimic_label = vindr_to_mimic_mapping[vindr_label]
            mimic_idx = MIMIC_LABELS.index(mimic_label)
            # Take OR: if any mapping is 1 → set 1
            if value == 1:
                mimic_row[mimic_idx] = 1
    return mimic_row


if __name__ == "__main__":
    """
        Main function Test on Vindr-cxr test dataset
    """
    print("Starting")
    print("Loading Graph JSON...")
    # Load the graph JSON file
    with open(MANUAL_GRAPH, 'r') as file:
        data_graph_json = json.load(file)

    # Check for device
    print("Checking for device...")
    t_device = torch.device("cpu" if torch.version.hip else
                            ("cuda" if torch.cuda.is_available() else "cpu"))
    is_cuda = torch.cuda.is_available() and not torch.version.hip
    # ROCm is not well-supported yet
    print(f"Using device: {t_device}")

    # Init Model
    med_model = SwinMIMICGraphClassifier(graph_json=data_graph_json, device=t_device).to(t_device)
    print("Model initialized.")

    print(" -- Verifying save directory and loading model state if exists --")
    SAVE_DIR = os.path.join(MODELS_DIR, "med-vix-ray")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # If it still doesn't exist exit, so avoid errors after waiting all training
    if not os.path.exists(SAVE_DIR):
        print("Unable to create save dir. Exiting...")
        exit(1)

    # Load Model if exists
    model_state_path = os.path.join(SAVE_DIR, "med_vixray_model_state.pth")
    json_graph_path = os.path.join(SAVE_DIR, "med_vixray_model_graph.json")

    if os.path.exists(model_state_path):
        print("[INFO] Model State found! Trying to load it...")
        try:
            with open(json_graph_path, 'r') as file:
                data_graph_json = json.load(file)
        except (FileNotFoundError, OSError):
            print("[WARNING] Graph JSON file not found.")
            exit(1)

        med_model.load_model_from_state(state_dict_path=model_state_path, graph_json=data_graph_json)
        print("[INFO] Model State loaded!")
    else:
        print("Unable to find model state... Exiting!")
        exit(1)

    # Directory with all DICOM files from ENV
    dicom_dir = os.getenv("VINDR_DIR", None)
    vindr_csv_path = os.getenv("VINDR_CSV_PATH", None)

    if not dicom_dir or not vindr_csv_path:
        print("Please set the VINDR_DIR and VINDR_CSV_PATH environment variables.")
        exit(1)

    # Read CSV VinDr
    vindr_df = pd.read_csv(vindr_csv_path)  # path to csv

    # Final columns MIMIC-style
    mimic_labels = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
                    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
                    "Lung Opacity","No Finding","Pleural Effusion","Pleural Other",
                    "Pneumonia","Pneumothorax","Support Devices"]

    # Create df with 1 row per image, 1 column for each of the 14 labels
    df_out = pd.DataFrame(0, index=vindr_df["image_id"].unique(), columns=mimic_labels)

    vindr_label_columns = list(vindr_to_mimic_mapping.keys())

    print(f"[INFO] VinDr columns number: {len(vindr_label_columns)}")
    print("Now mapping VinDr labels to MIMIC labels...")
    # Create new dataframe with MIMIC labels
    mimic_rows = []
    for _, row in vindr_df.iterrows():
        vindr_row = row[vindr_label_columns].to_dict()
        mimic_row = map_vindr_row_to_mimic(vindr_row)
        mimic_rows.append(mimic_row)

    df_out.loc[vindr_df["image_id"].values] = mimic_rows

    # Now df_out has shape (num_images, 14), compatibile with med-vix
    df_out.to_csv("vindr_test_mimic_labels.csv")
    print(f"Mapped labels saved to vindr_test_mimic_labels.csv with shape {df_out.shape}")

    # Create DataLoader
    # Create a list of (dicom_path, image_id) tuples
    dicom_paths = [(os.path.join(dicom_dir, f"{image_id}.dicom"), image_id) for image_id in df_out.index]
    vindr_dataset = VinDrDataset(dicom_paths, df_out, image_size=(256, 256))
    vindr_loader = DataLoader(vindr_dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"DataLoader created with {len(vindr_loader.dataset)} images.")

    # Run inference and evaluation for all images in the DataLoader
    med_model.eval()
    all_preds = []
    all_labels = []
    all_image_ids = []

    print("Starting inference on VinDr dataset...")
    with torch.no_grad():
        for x_batch, y_batch, image_ids in vindr_loader:
            x_batch = x_batch.to(t_device)
            outputs = med_model(x_batch)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
            all_image_ids.extend(image_ids)
            if len(all_preds) % 100 == 0:
                print(f"Processed {len(all_preds) * vindr_loader.batch_size} images so far...")

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Now all_preds, all_labels, and all_image_ids contain results for the whole dataset

    y_true = np.vstack(all_labels)  # shape [N, C]
    y_score = np.vstack(all_preds)  # shape [N, C]
    threshold = 0.5
    y_pred = (y_score > threshold).astype(int)

    # Macro and micro metrics
    metrics = {"Exact Match Ratio": accuracy_score(y_true, y_pred),
               "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
               "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
               "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
               "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
               "F1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
               "Precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
               "Recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0)}

    # ROC AUC e AUPRC per classe e media macro
    n_classes = y_true.shape[1]
    roc_aucs = []
    pr_aps = []
    # curve macro: concateniamo poi
    all_fpr = np.unique(np.linspace(0, 1, 100))
    mean_tpr = np.zeros_like(all_fpr)
    mean_prec = np.zeros_like(all_fpr)  # riutilizziamo xp per PR

    for c in range(n_classes):
        # ROC per classe
        fpr, tpr, _ = roc_curve(y_true[:, c], y_score[:, c])
        auc_c = roc_auc_score(y_true[:, c], y_score[:, c])
        roc_aucs.append(auc_c)
        # interp tpr su grid comune
        mean_tpr += np.interp(all_fpr, fpr, tpr)

        # PR per classe
        prec, rec, _ = precision_recall_curve(y_true[:, c], y_score[:, c])
        ap_c = average_precision_score(y_true[:, c], y_score[:, c])
        pr_aps.append(ap_c)
        # interp precision su stesso grid
        mean_prec += np.interp(all_fpr, rec[::-1], prec[::-1])

    # macro values
    metrics["ROC_AUC_macro"] = np.mean(roc_aucs)
    metrics["AUPRC_macro"] = np.mean(pr_aps)
    metrics["ROC_AUC_per_class"] = roc_aucs
    metrics["AUPRC_per_class"] = pr_aps

    # Save metrics to a JSON file
    with open("vindr_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to vindr_test_metrics.json")

    # Print Main results
    print("=== Main Metrics ===")
    for key, value in metrics.items():
        if not isinstance(value, list):
            print(f"{key}: {value:.4f}")



