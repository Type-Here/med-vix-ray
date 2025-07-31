import json
import os
import pickle
from typing import Optional, Union

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, recall_score, precision_score, \
    f1_score

from sklearn.model_selection import train_test_split
from torch.nn import Sequential, Linear, Dropout, ReLU
from torch.utils.data import DataLoader

from rsna_penumonia import load_pretrained_model, RSNADataset

from settings import BATCH_SIZE
from src.med_vix_ray import SwinMIMICGraphClassifier


def split_dataset(csv_bin_path: str) -> tuple[list[int], list[int], list[int]]:
    """
    Split the RSNA Pneumonia dataset into training, validation, and test sets.
    Creates split_train.pkl, split_val.pkl, split_test.pkl in the same directory as csv_bin_path.
    Uses a 60/20/20 split with stratification on 'Target'.
    ('Target' is column with either 0 or 1 indicating pneumonia presence)
    Split will be randomized using sklearn's train_test_split.

    :param csv_bin_path: Path to the binary CSV file containing dataset information.
    """
    rsna_df = pd.read_csv(csv_bin_path)
    rsna_df.drop(columns=[c for c in ['x', 'y', 'width', 'height'] if c in rsna_df], inplace=True)

    # Drop Duplicate Patient IDs
    rsna_df.drop_duplicates(subset=['patientId'], inplace=True)

    # Ensure 'Target' column exists
    if 'Target' not in rsna_df.columns:
        raise ValueError(f"'Target' column not found in {csv_bin_path}. Ensure the dataset has a 'Target' column.")
    # Check if 'Target' column is binary
    if not set(rsna_df['Target'].unique()).issubset({0, 1}):
        raise ValueError(f"'Target' column in {csv_bin_path} must be binary (0 or 1). Found unique values: {rsna_df['Target'].unique()}")


    # Prepare indices and labels
    indices = list(range(len(rsna_df)))
    labels = rsna_df['Target']

    # 60% train, 40% temp (to be split into val and test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=0.4, random_state=42, stratify=labels
    )
    # 20% val, 20% test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Ensure output directory exists
    split_dir = os.path.dirname(csv_bin_path)
    os.makedirs(split_dir, exist_ok=True)

    # Filter subset
    train_df = rsna_df.iloc[train_idx].reset_index(drop=True)
    val_df = rsna_df.iloc[val_idx].reset_index(drop=True)
    test_df = rsna_df.iloc[test_idx].reset_index(drop=True)

    # Save split datasets
    train_df.to_csv(os.path.join(split_dir, "split_train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "split_val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "split_test.csv"), index=False)

    return train_df, val_df, test_df

def manage_dataset():
    """
    Manage RSNA Pneumonia dataset splitting and loading.
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) where each is an instance of RSNADataset.
    """
    # Load RSNA Pneumonia dataset
    dicom_dir = os.getenv("RSNA_DIR", None)
    csv_bin_path = os.getenv("RSNA_CSV_PATH_BINARY", None)
    csv_dir = os.path.dirname(csv_bin_path)

    if dicom_dir is None or csv_bin_path is None:
        raise ValueError("Environment variables RSNA_DIR and RSNA_CSV_PATH_BINARY must be set.")

    # Search for already split dataset
    split_dataset_path = os.path.join(os.path.dirname(csv_bin_path), 'split_train.csv')

    if not os.path.exists(split_dataset_path):
        print(f"Split dataset not found at {split_dataset_path}. Creating new split...")
        train_df, val_df, test_df = split_dataset(csv_bin_path)
    else:
        print(f"Split dataset found at {split_dataset_path}. Loading existing split...")
        train_df = pd.read_csv(os.path.join(csv_dir, 'split_train.csv'))
        val_df = pd.read_csv(os.path.join(csv_dir, 'split_val.csv'))
        test_df = pd.read_csv(os.path.join(csv_dir, 'split_test.csv'))

    # Create complete paths for all files
    rsna_df = pd.read_csv(csv_bin_path)
    dicom_paths = [(os.path.join(dicom_dir, f"{patientID}.dcm"), patientID) for patientID in rsna_df['patientId']]

    # Create dcm_paths for each subset
    train_paths = [(os.path.join(dicom_dir, f"{row['patientId']}.dcm"), row['patientId']) for _, row in
                   train_df.iterrows()]
    val_paths = [(os.path.join(dicom_dir, f"{row['patientId']}.dcm"), row['patientId']) for _, row in val_df.iterrows()]
    test_paths = [(os.path.join(dicom_dir, f"{row['patientId']}.dcm"), row['patientId']) for _, row in
                  test_df.iterrows()]

    # Create dataset instances for train, validation, and test sets
    train_dt = RSNADataset(train_paths, train_df, image_size=(256, 256), is_only_binary=True)
    val_dt = RSNADataset(val_paths, val_df, image_size=(256, 256), is_only_binary=True)
    test_dt = RSNADataset(test_paths, test_df,  image_size=(256, 256), is_only_binary=True)
    return train_dt, val_dt, test_dt


def fine_tune_to_rsna(model_to_ft: SwinMIMICGraphClassifier, model_saving_path: str):
    """
    Fine-tune the SwinMIMICGraphClassifier model on the RSNA Pneumonia dataset.
    This function assumes the model is already loaded with pre-trained weights.
    :param model_to_ft: SwinMIMICGraphClassifier instance with pre-trained weights.
    :param model_saving_path: Path to save the fine-tuned model.
    """

    # Set Parameters
    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    # Optimizer
    optimizer = torch.optim.Adam(model_to_ft.classifier.parameters(), lr=1e-4)  # Learning rate for classifier head

    # Fine-tune the model: Create a training loop
    for epoch in range(5):
        model_to_ft.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(t_device)
            labels = labels.to(t_device).float()

            optimizer.zero_grad()  # Reset gradients before each batch

            # Forward pass
            train_outputs = model_to_ft(images)
            loss, _ = model_to_ft.compute_total_loss(train_outputs, labels, model_to_ft.signs_found, loss_fn)

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/5], Loss: {epoch_loss:.4f}")

    print("[INFO] Fine-tuning complete.")

    # Save the fine-tuned model
    model_to_ft.save_all(path=model_saving_path)

    print(f"[INFO] Fine-tuned model saved to {model_saving_path}")


def adapt_model_to_rsna(model_to_ft, device: Optional[Union[torch.device, str]] = "cpu"):
    """
    Adapt the pre-trained SwinMIMICGraphClassifier model to the RSNA Pneumonia dataset.
    This function freezes the Swin Transformer layers and replaces the classifier head with a new one suitable
    for binary classification (pneumonia detection).
    :param model_to_ft: SwinMIMICGraphClassifier instance with pre-trained weights.
    :param device: Device to move the model to (CPU or GPU).
    :raises ValueError: If the model is not a SwinMIMICGraphClassifier instance.
    """

    # Freeze all layers
    for param in model_to_ft.parameters():
        param.requires_grad = False
    model_to_ft.swin_model.requires_grad_(False)  # Freeze Swin Transformer layers
    # Remove Classifier head and add new one
    model_to_ft.classifier = Sequential(
        Linear(in_features=1024, out_features=512, bias=True),
        ReLU(),
        Dropout(p=0.3),
        Linear(in_features=512, out_features=1, bias=True)
    )
    model_to_ft.classifier.requires_grad_(True)  # Only train the new classifier head
    model_to_ft.train(True)
    model_to_ft.is_fine_tuning = True  # Set fine-tuning mode
    model_to_ft.num_classes = 1  # Set number of classes for binary classification for the graph output
    # Debug: updatable parameters
    for name, param in model_to_ft.named_parameters():
        if param.requires_grad:
            print(f"[DEBUG] Trainable parameter: {name}")
    model_to_ft.to(device)  # Move model to device


if __name__ == "__main__":
    """
        Main function to fine-tune Med-ViX-Ray model on RSNA Pneumonia dataset.
    """
    print(" ========= MAIN FINE-TUNE RSNA PNEUMONIA MODEL ========= ")

    # Load dataset
    train_dataset, val_dataset, test_dataset = manage_dataset()

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Try to load an already fine-tuned model
    model_save_path = os.path.join(os.path.dirname(__file__), 'rsna_finetuned_model_state.pth')
    json_graph_path = os.path.join(os.path.dirname(__file__), 'rsna_finetuned_model_graph.json')

    # Load pretrained model
    model, t_device = load_pretrained_model()

    # If model is already fine-tuned or not: adapt it to RSNA Pneumonia dataset
    # because only state_dict is saved, not the whole model
    adapt_model_to_rsna(model, t_device)

    if os.path.exists(model_save_path):
        print("[INFO] RSNA Fine-Tuned Model State found! Trying to load it...")
        try:
            with open(json_graph_path, 'r') as file:
                data_graph_json = json.load(file)
        except (FileNotFoundError, OSError):
            print("[WARNING] Graph JSON file not found. Using default graph.")
            exit(1)
        model.load_model_from_state(state_dict_path=model_save_path, graph_json=data_graph_json)
        print("[INFO] Fine-Tuned Model State loaded!")

    else:
        print("[INFO] Model State Fine-Tuned not found! Using default model.")
        print("[INFO] Fine-tuning the model on RSNA Pneumonia dataset...")
        fine_tune_to_rsna(model, model_save_path)

    # Any case: Evaluate the model

    # Evaluate the model on test set
    print(" --------- ")
    print("[INFO] Evaluating model on test set...")

    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        length = len(test_loader.dataset)
        count = 0

        # Iterate over test dataset
        for images, labels, _ in test_loader:
            images = images.to(t_device)
            labels = labels.to(t_device)

            outputs = model(images)
            # Convert logits to probabilities and then to binary predictions
            predictions = torch.sigmoid(outputs)

            # Flatten and extend scalars for correct metric inputs
            all_scores.extend(predictions.squeeze().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend((predictions > 0.5).squeeze().cpu().numpy().astype(int).tolist())

            count += len(images)
            if count % 100 == 0:
                print(f"[INFO] Processed {count}/{length} batches.")

    y_true = all_labels
    y_pred = all_predictions
    y_score = all_scores

    # Calculate Metrics
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

    # Save metrics to a file
    metrics_save_path = os.path.join(os.path.dirname(__file__), 'rsna_finetuned_metrics.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(bin_metrics, f, indent=4)
    print(f"[INFO] Evaluation metrics saved to {metrics_save_path}")

    # Done
    print("[INFO] Evaluation complete.")
    print(" --------- ")

    print("Exiting...")