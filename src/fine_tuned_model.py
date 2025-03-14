import logging

import torch
import timm
import os
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import dataset.dataset_handle as dh

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Dataset path
from settings import DATASET_PATH, MIMIC_LABELS, MODELS_DIR, DOWNLOADED_FILES, NUM_WORKERS
from settings import NUM_EPOCHS, UNBLOCKED_LEVELS, LEARNING_RATE_CLASSIFIER, LEARNING_RATE_TRANSFORMER
from settings import SWIN_MODEL_SAVE_PATH, SWIN_STATS_PATH

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocess import ImagePreprocessor


@logging.WARN("Not fully implemented yet")
def vit_loader():
    """
    Vision Transformer (ViT) src for feature extraction.
    ViT is a transformer-based src that processes images as sequences of patches.
    Warning
    -------
    This function is not fully implemented yet.
    """
    # Transformers for ViT
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # ViT accepts 256x256 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [0, 1] range in grayscale
    ])
    # Load dataset
    dataset = ImageFolder(root=DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    # Test dataloader
    images, labels = next(iter(dataloader))
    print(f"Shape batch immagini: {images.shape}")  # Output: [16, 1, 256, 256]
    # -------------------------------------------------------------------------------
    # Load pretrained ViT src
    vit_model = timm.create_model('vit_base_patch16_256', pretrained=True)
    # Substitute head for feature extraction
    vit_model.head = nn.Identity()  # Output embedding instead of classification
    # Test forward pass
    with torch.no_grad():
        vit_features = vit_model(images.repeat(1, 3, 1, 1))  # ViT requires RGB input (3 channels)
        print(f"Output ViT feature shape: {vit_features.shape}")


# ================================================ MIMIC SWIN CLASSIFIER ===============================================


class SwinMIMICClassifier(nn.Module):
    """
        SwinMIMICClassifier is a multi-label classification src based on the Swin V2 architecture.
        It is designed to classify images from the MIMIC-CXR dataset into multiple classes.
        The src consists of a Swin V2 feature extractor followed by a custom classifier head.

        Attributes:
          swin_model (nn.Module): The Swin V2 feature extractor.
          classifier (nn.Sequential): The custom classifier head for multi-label classification.

        :argument nn.Module: Inherits from PyTorch's nn.Module class.
    """

    def __init__(self, num_classes=len(MIMIC_LABELS)):  # 14 patologie in MIMIC-CXR
        """
            Initializes the SwinMIMICClassifier src.
            This src uses the Swin V2 architecture for feature extraction and a custom classifier
            head for multi-label classification.
            :param num_classes: Number of output classes (default: 14 for MIMIC-CXR).
        """
        super(SwinMIMICClassifier, self).__init__()
        self.swin_model = self.__swin_loader(num_classes=num_classes)

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # Swin V2 output: 1024-dim → reduced to 512
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout
            nn.Linear(512, num_classes),  # Final Output: 14 classes (MIMIC-CXR labels)
            nn.Sigmoid()  # Sigmoid activation for multi-label classification
        )

        # Move src to available device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __swin_loader(self, evaluation=False, num_classes=14):
        """
        Swin V2 src for feature extraction.
        Swin V2 is a hierarchical transformer that computes representation with shifted windows.
        It is designed to be efficient and effective for various vision tasks.

        Simplified Swin V2 Architecture
        ------------------------------
        1. Patch Embedding: Converts the image into patches by applying a convolutional layer with large kernel size.
        2. Transformer Encoder: Applies self-attention to the patches, allowing the src to learn relationships between them.
        3. MLP Head: A multi-layer perceptron that processes the output of the transformer encoder to produce the final feature representation.

        Args:
            evaluation (bool): set model directly in evaluation mode if Ture. Defaults to False
        """

        # Load Pre-Trained src
        model = timm.create_model("swinv2_base_window8_256.ms_in1k", pretrained=True, num_classes=num_classes)

        # Modify the first convolutional layer to accept grayscale input (1 channel) instead of RGB (3 channels)
        conv1 = model.patch_embed.proj  # First conv layer
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=(conv1.kernel_size[0], conv1.kernel_size[1]), # Check if correct
            stride=(conv1.stride[0], conv1.stride[1]), # Check if correct
            padding=conv1.padding,
            bias=(conv1.bias is not None)
        )

        # Copy the weights from the original conv layer, averaging across channels
        new_conv1.weight.data = conv1.weight.mean(dim=1, keepdim=True)

        # Replace the first conv layer in the src
        model.patch_embed.proj = new_conv1

        # Remove final classifier to extract features
        model.head = nn.Identity()

        # Set src to evaluation mode
        if evaluation:
            model.eval()

        # Test on a sample image
        # with torch.no_grad():
        #    features = swin_model(image_tensor)  # Pass the pre-processed image tensor
        #    print(f"Output Swin V2 feature shape: {features.shape}")  # Output: torch.Size([1, 1024])
        return model

    def forward(self, x):
        x = self.swin_model(x)  # Pass through the Swin V2 src
        x = self.classifier(x)  # Pass through the classifier head
        return x  # Output: probabilities for each class

    def train_model(self, train_loader, num_epochs=NUM_EPOCHS, learning_rate_swin=LEARNING_RATE_TRANSFORMER,
                    learning_rate_classifier=LEARNING_RATE_CLASSIFIER,
                    layers_to_unblock=UNBLOCKED_LEVELS, optimizer_param=None, loss_fn_param=nn.BCELoss):
        """
        Train the SwinMIMICClassifier src.
        This method trains the src using the provided training and validation data loaders.
        It unblocks the specified number of transformer blocks and the classifier head for training.
        The training process includes forward and backward passes, loss calculation, and optimizer step.
        The src is trained using the Adam optimizer with different learning rates for the Swin Transformer layers and the classifier head.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the src.
            learning_rate_classifier (float): Learning rate for the classifier head.
            learning_rate_swin (float): Learning rate for the Swin Transformer layers.
            optimizer_param (torch.optim.Optimizer): Optimizer for training. Default: None -> optim.Adam will be used.
            loss_fn_param (torch.nn.Module): Loss function for training. Default: nn.BCELoss.
            layers_to_unblock (int): Number of transformer blocks to unblock for training.
        """

        # Freeze all layers initially
        for param in self.swin_model.swin.parameters():
            param.requires_grad = False

        # Unfreeze the last layers_to_unblock transformer blocks # Default: 3
        for layer in list(self.swin_model.swin.layers)[-layers_to_unblock:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze the classifier head
        for param in self.swin_model.classifier.parameters():
            param.requires_grad = True

        # List all parameters and their requires_grad status (whether they are trainable)
        for name, param in self.swin_model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

        # Define two parameter groups with different learning rates
        # Group 1: Swin Transformer layers
        # Group 2: Classifier head
        if optimizer_param is None:
            optimizer = optim.Adam([
                {"params": self.swin_model.swin.layers[-3:].parameters(), "lr": learning_rate_swin}, # Lower LR for Swin Transformer
                {"params": self.swin_model.classifier.parameters(), "lr": learning_rate_classifier}  # Higher LR for classifier head
            ])
        else:
            optimizer = optimizer_param

        # Binary Cross-Entropy for multi-label classification
        loss_fn = loss_fn_param

        for epoch in range(num_epochs):
            self.swin_model.train()
            running_loss = 0.0

            for images, labels in train_loader:

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = self.swin_model(images)
                loss = loss_fn(outputs, labels)

                # Backpropagation
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

    def model_evaluation(self, val_loader, threshold=0.5, save_stats=True):
        """
        Evaluate the src using scikit-learn metrics for multi-label classification.
        Args:
            val_loader: DataLoader for validation data.
            threshold: Threshold for binary classification.
            save_stats: If True, save the evaluation metrics to a file.
        Returns:
            Dictionary with evaluation metrics.
        """
        self.eval()
        self.swin_model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self(images)

                # Convert probabilities to binary predictions
                predicted = (outputs > threshold).cpu().numpy()
                labels = labels.cpu().numpy()

                all_preds.append(predicted)
                all_labels.append(labels)

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate metrics
        metrics = {
            "Exact Match Ratio (EMR)": accuracy_score(all_labels, all_preds), # % of images with ALL labels correct
            "F1-score (macro)": f1_score(all_labels, all_preds, average="macro"),
            "F1-score (weighted)": f1_score(all_labels, all_preds, average="weighted"),
            "Precision (macro)": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "Recall (macro)": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "ROC AUC (macro)": roc_auc_score(all_labels, all_preds, average="macro"),
        }

        # Print metrics
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save metrics to file
        if save_stats:
            with open(SWIN_STATS_PATH, 'w') as file:
                file.write(str(metrics))
            print(f"Metrics saved to {SWIN_STATS_PATH}")

        return metrics

    def save_model(self, path=SWIN_MODEL_SAVE_PATH):
        """
        Save the trained src to a file.
        Args:
            path (str): Path to save the src. Default: SWIN_MODEL_SAVE_PATH.
        """
        torch.save(self.swin_model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path=SWIN_MODEL_SAVE_PATH):
        """
        Load a trained src from a file.
        Args:
            path (str): Path to load the src from. Default: SWIN_MODEL_SAVE_PATH.
        """
        if not path.endswith('.pth'):
            raise ValueError("Path must end with .pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load the src state
        self.swin_model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


if __name__ == "__main__":

    # Initialize the SwinMIMICClassifier
    ft_model = SwinMIMICClassifier()

    SAVE_DIR = os.path.join(MODELS_DIR, "fine_tuned")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load Model if exists
    model_path = os.path.join(SAVE_DIR, "fine_tuned_model.pth")

    if os.path.exists(model_path):
        print(f"Model exists in {model_path}; Load it?")
        if input("y/N: ").lower() == "y":
            ft_model.load_model(model_path)
        elif input("Do you want to train a new model? (y/N): ").lower() == "y":
            pass
        else:
            print("Exiting...")
            exit(0)

    # Load dataset
    try:
        train_dataset = dh.load_ready_dataset(phase='train')
    except FileNotFoundError:
        print("Train dataset not found. Creating a new one.")
        merged_data = dh.dataset_handle(partial_list=DOWNLOADED_FILES)
        train_dataset, _, _ = dh.split_dataset(merged_data)

    try:
        validation_dataset = dh.load_ready_dataset(phase='validation')
    except FileNotFoundError:
        print("Validation dataset not found. Since Train dataset was created or loaded \n there should be a validation dataset too.")
        print("Check the dataset folder or code.")
        exit(1)

    print("Train and Validation information dataset loaded.")

    # Obtain Paths
    train_image_paths = dh.fetch_image_from_csv(train_dataset, DATASET_PATH)
    val_image_paths = dh.fetch_image_from_csv(validation_dataset, DATASET_PATH)

    # Obtain Dataloaders in order to improve performance
    training_loader = DataLoader(ImagePreprocessor(train_image_paths), batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(ImagePreprocessor(val_image_paths), batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    # Train the model
    print("Starting training...")
    # NOTE: for other parameters, settings.py defines default values
    ft_model.train_model(training_loader)

    # Save the model
    print("Saving model...")
    ft_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model
    print("Starting evaluation...")
    metrics_dict = ft_model.model_evaluation(valid_loader, save_stats=False)
    print("Evaluation completed.")
    print("Metrics:", metrics_dict)

    # Save the metrics to a file
    metrics_file = os.path.join(SAVE_DIR, "finte_tuned_metrics.json")
    with open(metrics_file, 'w') as f:
        f.write(str(metrics_dict))
    print(f"Metrics saved to {metrics_file}")

    # Save the model architecture to a file
    model_architecture_file = os.path.join(SAVE_DIR, "model_architecture.txt")
    with open(model_architecture_file, 'w') as f:
        f.write(str(ft_model))
    print(f"Model architecture saved to {model_architecture_file}")

    print("Training and evaluation completed.")
    print("Exiting...")
