import json
import os
import pickle
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, recall_score, precision_score, \
    f1_score

from sklearn.model_selection import train_test_split
from torch.nn import Sequential, Linear, Dropout, ReLU
from torch.utils.data import DataLoader

from rsna_penumonia import load_pretrained_model, RSNADataset

from settings import BATCH_SIZE

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

    # Save splits
    for name, split in [('split_train.pkl', train_idx), ('split_val.pkl', val_idx), ('split_test.pkl', test_idx)]:
        path = os.path.join(split_dir, name)
        with open(path, 'wb') as f:
            pickle.dump(split, f)

    print(f"[INFO] Dataset split completed and saved to {split_dir}")
    return train_idx, val_idx, test_idx

def manage_dataset():
    """
    Manage RSNA Pneumonia dataset splitting and loading.
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) where each is an instance of RSNADataset.
    """
    # Load RSNA Pneumonia dataset
    dicom_dir = os.getenv("RSNA_DIR", None)
    csv_bin_path = os.getenv("RSNA_CSV_PATH_BINARY", None)

    if dicom_dir is None or csv_bin_path is None:
        raise ValueError("Environment variables RSNA_DIR and RSNA_CSV_PATH_BINARY must be set.")

    # Search for already split dataset
    # From csv_bin_path, change the file name to 'split_train.pkl'
    split_dataset_path = os.path.join(os.path.dirname(csv_bin_path), 'split_train.pkl')
    if not os.path.exists(split_dataset_path):
        print(f"Split dataset not found at {split_dataset_path}. Creating new split...")
        train_indices, val_indices, test_indices = split_dataset(csv_bin_path)
    else:
        print(f"Split dataset found at {split_dataset_path}. Loading existing split...")
        with open(split_dataset_path, 'rb') as f:
            train_indices = pickle.load(f)
        with open(os.path.join(os.path.dirname(csv_bin_path), 'split_val.pkl'), 'rb') as f:
            val_indices = pickle.load(f)
        with open(os.path.join(os.path.dirname(csv_bin_path), 'split_test.pkl'), 'rb') as f:
            test_indices = pickle.load(f)

    # Create dataset instances for train, validation, and test sets
    train_dataset = RSNADataset(dicom_dir, csv_bin_path, train_indices)
    val_dataset = RSNADataset(dicom_dir, csv_bin_path, val_indices)
    test_dataset = RSNADataset(dicom_dir, csv_bin_path, test_indices)
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    """
        Main function to fine-tune Med-ViX-Ray model on RSNA Pneumonia dataset.
    """

    # Load pretrained model
    model, t_device = load_pretrained_model()

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    model.swin_model.requires_grad_(False)  # Freeze Swin Transformer layers

    # Remove Classifier head and add new one
    model.classifier = Sequential(
        Linear(in_features=1024, out_features=512, bias=True),
        ReLU(),
        Dropout(p=0.3),
        Linear(in_features=512, out_features=1, bias=True)
    )

    model.classifier.requires_grad_(True)  # Only train the new classifier head
    model.train(True)
    model.is_fine_tuning = True  # Set fine-tuning mode

    # Debug: updatable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[DEBUG] Trainable parameter: {name}")

    model.to(t_device)  # Move model to device

    # Load dataset
    train_dataset, val_dataset, test_dataset = manage_dataset()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Set Parameters
    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    # Optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)  # Learning rate for classifier head

    # Fine-tune the model: Create a training loop
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(t_device)
            labels = labels.to(t_device).float().unsqueeze(1)  # (B,) -> (B,1)


            optimizer.zero_grad()  # Reset gradients before each batch

            # Forward pass
            outputs = model(images)
            loss, _ = model.compute_total_loss(outputs, labels, model.signs_found, loss_fn)

            loss.backward()        # Backward pass
            optimizer.step()       # Update weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/5], Loss: {epoch_loss:.4f}")

    print("[INFO] Fine-tuning complete.")

    # Save the fine-tuned model
    model_save_path = os.path.join(os.path.dirname(__file__), 'rsna_finetuned_model.pth')
    model.save_all(path=model_save_path)

    print(f"[INFO] Fine-tuned model saved to {model_save_path}")

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
        length = len(test_loader)
        count = 0

        # Iterate over test dataset
        for images, labels in test_loader:
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