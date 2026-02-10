import json
import os
from typing import Tuple

import pydicom
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from settings import MANUAL_GRAPH, MODELS_DIR
from src.med_vix_ray import SwinMIMICGraphClassifier

# Load the pre-trained model
def load_pretrained_model() -> Tuple[SwinMIMICGraphClassifier, torch.device]:
    """
        Load the pre-trained SwinMIMICGraphClassifier model and its graph JSON.
        Returns: Tuple: SwinMIMICGraphClassifier, torch.device
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
        return med_model, t_device
    else:
        print("Unable to find model state... Exiting!")
        exit(1)


class RSNADataset(torch.utils.data.Dataset):
    def __init__(self, dcm_paths, df_out, image_size=(256, 256), is_only_binary=False):
        self.dicom_paths:list = dcm_paths
        self.df_out:DataFrame = df_out
        self.image_size:Tuple = image_size
        self.is_only_binary = is_only_binary

    def __len__(self):
        return len(self.dicom_paths)

    def __getitem__(self, idx):
        dicom_path, image_id = self.dicom_paths[idx]
        x = preprocess_rsna_dicom(dicom_path, self.image_size)
        row = self.df_out[self.df_out['patientId'] == image_id].iloc[0]
        if self.is_only_binary:
            # For binary classification, we only need the 'Target' column
            y = torch.tensor([row['Target']], dtype=torch.float32)
        else:
            y = torch.tensor([row['Target'], row['ternary']], dtype=torch.float32)
        return x.squeeze(0), y, image_id



def preprocess_rsna_dicom(dicom_path, image_size=(256, 256)):
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


# MIMIC labels in order (needed for index lookup)
# MIMIC_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
#                "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
#                "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
#                "Pneumonia", "Pneumothorax", "Support Devices"]  # Support Devices → no match in VINDR


# ================= MAIN FUNCTION =================

if __name__ == "__main__":
    """
        Main function Test on RSNA-Pnemonia test dataset
    """
    # Load the pre-trained model
    med_model, t_device = load_pretrained_model()

    # Directory with all DICOM files from ENV
    dicom_dir = os.getenv("RSNA_DIR", None)
    rsna_csv_path_binary = os.getenv("RSNA_CSV_PATH_BINARY", None)
    rsna_csv_path_ternary = os.getenv("RSNA_CSV_PATH_TERN", None)

    if not dicom_dir or not rsna_csv_path_binary or not rsna_csv_path_ternary:
        print("Please set the RSNA_DIR and RSNA_CSV_PATH_{BINARY, TERN} environment variables.")
        exit(1)

    # Read CSV rsna
    rsna_df = pd.read_csv(rsna_csv_path_binary)  # path to csv
    rsna_df_tern = pd.read_csv(rsna_csv_path_ternary)  # path to csv ternary

    # Merge the two dataframes on same 'patientId':
    # From ternary read 'class' column: map it to values:
    # if value is 'No Lung Opacity / Not Normal' -> 1
    # if value is 'Lung Opacity' -> 2 (Pneumonia)
    # else if value is 'Normal' -> 0
    rsna_df = rsna_df.merge(rsna_df_tern[["patientId", "class"]], on="patientId", how="left")
    rsna_df = rsna_df.rename(columns={"class": "ternary"})

    rsna_df["ternary"] = rsna_df["ternary"].map({
        "No Lung Opacity / Not Normal": 1,
        "Lung Opacity": 2,
        "Normal": 0
    })

    # Drop 'x', 'y', 'width', 'height' columns if they exist
    rsna_df.drop(columns=["x", "y", "width", "height"], errors='ignore', inplace=True)

    # Remove duplicates based on 'patientId' and keep the first occurrence
    rsna_df = rsna_df.drop_duplicates(subset=["patientId"])

    # Get Random 5337 image rows for test set
    rsna_df = rsna_df.sample(n=5337, random_state=42).reset_index(drop=True)

    # Save the dataframe to a CSV file
    rsna_df.to_csv("rsna_test_data.csv", index=False)
    print("[INFO] RSNA DataFrame saved to rsna_test_data.csv")

    # Print shape and first few rows of the dataframe
    print(f"[INFO] VinDr DataFrame shape: {rsna_df.shape}")
    print("[INFO] VinDr DataFrame head:")
    print(rsna_df.head())




    # Final columns MIMIC-style
    mimic_labels = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
                    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
                    "Lung Opacity","No Finding","Pleural Effusion","Pleural Other",
                    "Pneumonia","Pneumothorax","Support Devices"]


    # Create DataLoader
    # Create a list of (dicom_path, image_id) tuples
    dicom_paths = [(os.path.join(dicom_dir, f"{patientID}.dcm"), patientID) for patientID in rsna_df['patientId']]
    rsna_dataset = RSNADataset(dicom_paths, rsna_df, image_size=(256, 256))
    rsna_loader = DataLoader(rsna_dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"DataLoader created with {len(rsna_loader.dataset)} images.")

    # Run inference and evaluation for all images in the DataLoader
    med_model.eval()
    bin_preds = []
    tern_preds = []
    bin_labels = []
    tern_labels = []
    all_image_ids = []
    bin_scores = []
    tern_scores = []

    print("[INFO] Starting inference on RSNA-Pneumonia dataset...")
    with torch.no_grad():
        for x_batch, y_batch, image_ids in rsna_loader:
            x_batch = x_batch.to(t_device)
            outputs = med_model(x_batch)
            preds = torch.sigmoid(outputs).cpu().numpy()

            score = np.maximum(preds[:, 11], preds[:, 7], preds[:, 2])  # Pneumonia, Lung Opacity, or Consolidation
            bin_scores.append(score.reshape(-1, 1))  # Binary scores for Pneumonia

            # Map predictions to RSNA: create a single data with values:
            # No Findings, preds[8] -> 0
            # Pneumonia, preds[11], pred[7] (Lung Opacity) -> 2
            # Other: The rest of the labels -> 1
            ternary_pred = np.zeros((preds.shape[0], 1), dtype=np.float32)
            for i in range(preds.shape[0]):
                if preds[i, 8] > 0.5:
                    ternary_pred[i, 0] = 0
                elif preds[i, 11] > 0.5 or preds[i, 7] > 0.5 or preds[i, 2] > 0.5:  # Pneumonia or Lung Opacity
                    ternary_pred[i, 0] = 2
                else:
                    ternary_pred[i, 0] = 1

            binary_preds = np.zeros((preds.shape[0], 1), dtype=np.float32)
            for i in range(preds.shape[0]):
                # Map ternary_pred to binary_preds
                if ternary_pred[i, 0] == 0 or ternary_pred[i, 0] == 1: # No Findings or Other
                    binary_preds[i, 0] = 0
                else: binary_preds[i, 0] = 1  # Pneumonia

            # Append predictions and labels
            bin_preds.append(binary_preds)
            tern_preds.append(ternary_pred)

            # From y_batch, we need to extract the labels for binary and ternary,
            # 'Target' and 'ternary' columns respectively
            bin_labels.append(y_batch.squeeze(1)[:, 0].numpy().reshape(-1, 1))  # Target is at index 0
            tern_labels.append(y_batch.squeeze(1)[:, 1].numpy().reshape(-1, 1))  # ternary is at index 1

            # Append image IDs
            all_image_ids.extend(image_ids)

            # Print progress every 100 images
            if len(bin_preds) % 100 == 0:
                print(f"Processed {len(bin_preds) * rsna_loader.batch_size} images so far...")

    # Concatenate results
    bin_preds = np.concatenate(bin_preds, axis=0)
    tern_preds = np.concatenate(tern_preds, axis=0)

    bin_scores = np.concatenate(bin_scores, axis=0)

    bin_labels = np.concatenate(bin_labels, axis=0)
    tern_labels = np.concatenate(tern_labels, axis=0)

    # Now all_preds, all_labels, and all_image_ids contain results for the whole dataset

    # Calculate metrics for binary classification
    y_true = bin_labels.flatten()
    y_pred = bin_preds.flatten() # Already binary: 0 or 1
    y_score = bin_scores.flatten() # Scores for Pneumonia (0-1)
    print("Calculating binary classification metrics...")

    # Save y_true, y_pred, y_score for binary classification
    with open("rsna_binary_test_results.json", "w") as f:
        json.dump({
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
            "image_ids": all_image_ids
        }, f, indent=2)
    print("Binary classification results saved to rsna_binary_test_results.json")

    # Binary metrics
    bin_metrics = {
        "Exact Match Ratio": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "ROC_AUC_micro": roc_auc_score(y_true, y_score, average="micro"),
        "ROC_AUC_macro": roc_auc_score(y_true, y_score, average="macro"),
        "AUPRC_micro": average_precision_score(y_true, y_score, average="micro"),
        "AUPRC_macro": average_precision_score(y_true, y_score, average="macro")
    }

    # Save binary metrics to a JSON file
    with open("rsna_binary_test_metrics.json", "w") as f:
        json.dump(bin_metrics, f, indent=2)
    print("Binary metrics saved to rsna_binary_test_metrics.json")

    # Ternary classification metrics
    print("Calculating ternary classification metrics...")
    y_true = tern_labels.flatten()
    y_pred = tern_preds.flatten()  # Already ternary: 0, 1, or 2
    y_score = tern_preds.flatten()  # Use the same for score, since it's ternary

    ternary_metrics = {"Exact Match Ratio": accuracy_score(y_true, y_pred),
               "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
               "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
               "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
               "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
               "F1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
               "Precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
               "Recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0)
               }

    # Save ternary metrics to a JSON file
    with open("rsna_ternary_test_metrics.json", "w") as f:
        json.dump(ternary_metrics, f, indent=2)
    print("Ternary metrics saved to rsna_ternary_test_metrics.json")

    print("Inference and evaluation completed successfully!")



