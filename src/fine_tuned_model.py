import json
#import sys

import torch
import timm
import os
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset path
from settings import DATASET_PATH, MIMIC_LABELS, MODELS_DIR, SWIN_MODEL_DIR, LEARNING_RATE_INPUT_LAYER
from settings import NUM_EPOCHS, UNBLOCKED_LEVELS, LEARNING_RATE_CLASSIFIER, LEARNING_RATE_TRANSFORMER
from settings import SWIN_MODEL_SAVE_PATH

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, average_precision_score

from src import general


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
def _swin_loader_rgb(evaluation: bool=False, num_classes: int=0) -> nn.Module:
    """
    Carica SwinV2 pretrained ms_in1k, senza toccare patch_embed
    e senza classifier head (num_classes=0 + head=Identity).
    """
    model = timm.create_model(
        "swinv2_base_window8_256.ms_in1k",
        pretrained=True,
        num_classes=num_classes,
    )

    # Remove final classifier to extract features
    model.head = nn.Identity() # Identity() removes the classifier head by replacing it with an identity function
    if evaluation:
        model.eval()
    return model

def _swin_loader(evaluation=False, num_classes=14) -> nn.Module:
    """
    Swin V2 src for feature extraction.
    Swin V2 is a hierarchical transformer that computes representation with shifted windows.
    It is designed to be efficient and effective for various vision tasks.

    Simplified Swin V2 Architecture
    ------------------------------
    1. Patch Embedding: Converts the image into patches by applying
    a convolutional layer with large kernel size. \n
    2. Transformer Encoder: Applies self-attention to the patches,
    allowing the src to learn relationships between them. \n
    3. MLP Head: A multi-layer perceptron that processes the output of the transformer encoder
    to produce the final feature representation.

    Args:
        evaluation (bool): set model directly in evaluation mode if Ture. Defaults to False
    Returns:
        model (nn.Module): The Swin V2 src for feature extraction.
    """

    # Load Pre-Trained src
    model = timm.create_model("swinv2_base_window8_256.ms_in1k", pretrained=True, num_classes=0)
    # num_classes=0 should remove the head

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

    # Copy the bias if it exists
    new_conv1.bias.data = conv1.bias.data

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

    def __init__(self, device=None, num_classes=len(MIMIC_LABELS)):  # 14 patologie in MIMIC-CXR
        """
            Initializes the SwinMIMICClassifier src.
            This src uses the Swin V2 architecture for feature extraction and a custom classifier
            head for multi-label classification.
            :param num_classes: Number of output classes (default: 14 for MIMIC-CXR).
        """
        super(SwinMIMICClassifier, self).__init__()
        self.swin_model = _swin_loader_rgb(num_classes=num_classes)

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # Swin V2 output: 1024-dim → reduced to 512
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout
            nn.Linear(512, num_classes),  # Final Output: 14 classes (MIMIC-CXR labels)
            #nn.Sigmoid()  # Sigmoid activation for multi-label classification;
            # Sigmoid(): Not recommended if using BCEWithLogitsLoss
        )

        # Move src to available device (CPU/GPU) # Save device but not change it since instability issues on AMD
        if device:
            self.device = device
        else:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                print(f"Error moving model to device: {e}")
                exit(1)

    def forward(self, x, use_classifier=True):
        # Get features from the backbone (e.g., shape [B, 1024, 8, 8])
        features = self.swin_model.forward_features(x)
        # Printed the shape: [16, 8, 8, 1024] it needs to be permuted to [16, 1024, 8, 8]

        # Permute dimensions to get (B, C, H, W)
        features = features.permute(0, 3, 1, 2)  # Now shape: [16, 1024, 8, 8]

        # Check for NaN or Inf in input tensor
        #if torch.isnan(features).any() or torch.isinf(features).any():
        #    raise ValueError("FEATURES contains NaN or Inf values.")

        # Pooling to get [B, 1024, 1, 1]
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))

        # Flatten to get [B, 1024]
        pooled = pooled.view(pooled.size(0), -1)

        # Pass the pooled features through the classifier head
        if use_classifier:
            logits = self.classifier(pooled)
        else:
            # If not using the classifier, return the pooled features
            logits = pooled
        return logits

    def train_model(self, train_loader, num_epochs=NUM_EPOCHS,
                    learning_rate_swin=LEARNING_RATE_TRANSFORMER,
                    learning_rate_classifier=LEARNING_RATE_CLASSIFIER,
                    learning_rate_input_layer=LEARNING_RATE_INPUT_LAYER,
                    layers_to_unblock=UNBLOCKED_LEVELS, optimizer_param=None,
                    loss_fn_param= nn.BCEWithLogitsLoss()):
        """
        Train the SwinMIMICClassifier src.
        This method trains the src using the provided training and validation data loaders.
        It unblocks the specified number of transformer blocks and the classifier head for training.
        The training process includes forward and backward passes, loss calculation, and optimizer step.
        The src is trained using the Adam optimizer with different learning rates
        for the Swin Transformer layers and the classifier head.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train the src.
            learning_rate_classifier (float): Learning rate for the classifier head.
            learning_rate_swin (float): Learning rate for the Swin Transformer layers.
            learning_rate_input_layer (float): Learning rate for the input layer. (conv + norm)
            optimizer_param (torch.optim.Optimizer): Optimizer for training. Default: None -> optim.Adam will be used.
            loss_fn_param (torch.nn.Module): Loss function for training.
                Default: nn.BCEWithLogitsLoss() since mix sigmoid and BCE, recommended for multi-label classification.
            layers_to_unblock (int): Number of transformer blocks to unblock for training.
        """

        self._unblock_layers(layers_to_unblock)

        # List all parameters and their requires_grad status (whether they are trainable)
        for name, param in self.swin_model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

        optimizer = self._create_optimizer(layers_to_unblock, learning_rate_classifier,
                                           learning_rate_swin, learning_rate_input_layer,
                                           optimizer_param)

        # Binary Cross-Entropy for multi-label classification
        loss_fn = loss_fn_param

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            count = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = self(images)
                # No need to apply sigmoid since BCEWithLogitsLoss combines a sigmoid layer
                # and the BCELoss in one single class
                loss = loss_fn(outputs, labels)

                # Backpropagation
                loss.backward()
                optimizer.step()

                running_loss += loss.detach().item()

                if count % 1000 == 0:
                    print("Step:", count ," overall steps:", len(train_loader))
                count += 1

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
            print("Saving model...")
            # Save the model state after each epoch
            self.save_model_state(path=SWIN_MODEL_DIR)
            # Save the model after each epoch
            #self.save_model(path=SWIN_MODEL_SAVE_PATH)
            print(f"Model saved")



    def _create_optimizer(self, layers_to_unblock, learning_rate_classifier,
                          learning_rate_swin, learning_rate_input_layer, optimizer_param):
        # Define two parameter groups with different learning rates
        # Group 1: Swin Transformer layers
        # Group 2: Classifier head
        if optimizer_param is None:
            optimizer = optim.Adam([
                {"params": self.swin_model.patch_embed.proj.parameters(),
                 "lr": learning_rate_input_layer},
                {"params": self.swin_model.patch_embed.norm.parameters(),
                 "lr": learning_rate_input_layer},
                {"params": self.swin_model.layers[-layers_to_unblock:].parameters(),
                 "lr": learning_rate_swin},  # Lower LR for Swin Transformer
                {"params": self.classifier.parameters(),
                 "lr": learning_rate_classifier}  # Higher LR for classifier head
            ])
        else:
            optimizer = optimizer_param
        return optimizer

    def _unblock_layers(self, layers_to_unblock):
        # Freeze all layers initially
        for param in self.swin_model.parameters():
            param.requires_grad = False
        # Unfreeze the input layer since it was changed to accept grayscale input
        for param in self.swin_model.patch_embed.proj.parameters():
            param.requires_grad = True
        # Unfreeze the normalization layer
        for p in self.swin_model.patch_embed.norm.parameters():
            p.requires_grad = True
        # Unfreeze the last layers_to_unblock transformer blocks # Default: 2
        for layer in list(self.swin_model.layers)[-layers_to_unblock:]:
            # print(f"Unblocking layer: {layer}")
            for param in layer.parameters():
                param.requires_grad = True
        # self.swin_model.head = self.classifier  # Set the classifier head
        # Unfreeze the head
        # for param in self.swin_model.head.parameters():
        #    param.requires_grad = True

    def model_evaluation(self, testing_loader, threshold: float = 0.5,
                         save_stats: bool = True, out_dir: str = None):
        """
          Evaluate the model on a dataloader, calculate multilabel metrics, and plot:
            - ROC curves (macro)
            - Precision-Recall curves (macro)
          Saves both the values and the plots.

          Args:
              testing_loader (DataLoader): DataLoader for testing/validation.
              threshold (float): Threshold to binarize predictions.
              save_stats (bool): If True, saves metrics and plots to disk.
              out_dir (str): Directory to save the files.

          Returns:
              dict: All calculated metrics + paths to the plots.
          """
        # Default to current folder + evaluation if None
        if out_dir is None:
            # Find current folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.join(current_dir, "test_results")

        os.makedirs(out_dir, exist_ok=True)
        self.eval()
        self.swin_model.eval()

        all_labels = []
        all_scores = []

        # Number of batches
        num_batches = len(testing_loader)

        with torch.no_grad():
            for images, labels in testing_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels.cpu().numpy())

                if len(all_labels) % 200 == 0:
                    print("Step:", len(all_labels), "overall steps:", num_batches)
                # Early exit for debugging
                if len(all_labels) > 100:
                    break

        y_true = np.vstack(all_labels)  # shape [N, C]
        y_score = np.vstack(all_scores)  # shape [N, C]
        y_pred = (y_score > threshold).astype(int)

        # Metriche di base
        metrics = {
            "Exact Match Ratio": accuracy_score(y_true, y_pred),
            "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

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
        if save_stats:
            self._save_stats_improved(all_fpr, mean_prec, mean_tpr,
                                      metrics, n_classes, out_dir)

        # Stampa a video
        for k, v in metrics.items():
            # evita di stampare intere liste
            if isinstance(v, list):
                print(f"{k}: [see per-class values]")
            else:
                print(f"{k}: {v:.4f}")

        return metrics

    def _per_label_roc_pr_delta(self, y_true: np.ndarray,
                               y_score: np.ndarray,
                               label_names: list[str]
                               ) -> dict[str, dict[str, float]]:
        """
        Calculate ROC AUC, AUPRC and their difference (delta) for each label in a multi-label classification task.
        Args:
            y_true (np.ndarray): array binario [N, C] delle vere etichette.
            y_score (np.ndarray): array [N, C] delle probabilità previste.
            label_names (list[str]): nomi delle C etichette, in ordine.

        Returns:
            dict: {
               label_name: {
                 "roc_auc": float,
                 "auprc": float,
                 "delta": float # auprc - roc_auc
               },
               ...
            }
        """
        assert y_true.shape == y_score.shape, "Shapes di y_true e y_score devono coincidere"
        n_classes = y_true.shape[1]
        assert len(label_names) == n_classes, "Numero di nomi etichette diverso da C"

        results = {}
        for i, name in enumerate(label_names):
            # se la classe ha solo zeri o solo uni, roc_auc_score fallisce;
            # in quel caso impostiamo a np.nan
            try:
                roc = roc_auc_score(y_true[:, i], y_score[:, i])
            except ValueError:
                roc = float("nan")
            try:
                pr = average_precision_score(y_true[:, i], y_score[:, i])
            except ValueError:
                pr = float("nan")

            results[name] = {
                "roc_auc": roc,
                "auprc": pr,
                "delta": pr - roc
            }
        return results

    def _save_stats_improved(self, all_fpr, mean_prec, mean_tpr, metrics,
                             n_classes, out_dir):
        """
        Save the evaluation statistics and plots.
        Args:
            all_fpr (np.ndarray): All false positive rates for ROC curve.
            mean_prec (np.ndarray): Mean precision for PR curve.
            mean_tpr (np.ndarray): Mean true positive rate for ROC curve.
            metrics (dict): Dictionary of evaluation metrics.
            n_classes (int): Number of classes.
            out_dir (str): Output directory to save the plots and stats.
        """
        # Plot and Save ROC and PR curves

        # ROC curve macro
        mean_tpr /= n_classes
        plt.figure()
        plt.plot(all_fpr, mean_tpr, label=f"macro ROC (AUC={metrics['ROC_AUC_macro']:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (macro)")
        plt.legend()
        roc_path = os.path.join(out_dir, "roc_macro.png")
        plt.savefig(roc_path)
        plt.close()

        # Precision-Recall macro
        mean_prec /= n_classes
        plt.figure()
        plt.plot(all_fpr, mean_prec, label=f"macro PR (AUPRC={metrics['AUPRC_macro']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (macro)")
        plt.legend()
        pr_path = os.path.join(out_dir, "pr_macro.png")
        plt.savefig(pr_path)
        plt.close()

        # Save in JSON
        stats_path = os.path.join(out_dir, "metrics.json")
        with open(stats_path, "w") as fi:
            json.dump(metrics, fi, indent=2)
        print(f"Saved metrics to {stats_path}")
        print(f"Saved ROC plot to {roc_path}")
        print(f"Saved PR plot to  {pr_path}")

    def save_model(self, path=SWIN_MODEL_SAVE_PATH):
        """
        Save the trained src to a file.
        Args:
            path (str): Path to save the src. Default: SWIN_MODEL_SAVE_PATH.
        """
        torch.save(self, path)
        print(f"Model saved to {path}")

    def save_model_state(self, path=SWIN_MODEL_DIR):
        """
        Save the trained src state to a file.
        Args:
            path (str): Path to save the src. Default: SWIN_MODEL_SAVE_PATH.
        """
        path = os.path.join(path, "finetuned_model_state.pth")
        torch.save(self.state_dict(), path)
        print(f"Model state saved to {path}")

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
        self.load_state_dict(
            torch.load(path,
                       map_location="cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    print("Starting ...")

    # Check for device
    print("Checking for device...")
    t_device = torch.device("cpu" if torch.version.hip else
                            ("cuda" if torch.cuda.is_available() else "cpu"))
    is_cuda = torch.cuda.is_available() and not torch.version.hip
    # ROCm is not supported yet
    print(f"Using device: {t_device}")

    # Initialize the SwinMIMICClassifier
    ft_model = SwinMIMICClassifier(device=t_device).to(t_device)

    SAVE_DIR = os.path.join(MODELS_DIR, "fine_tuned")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load Model if exists
    model_path = os.path.join(SAVE_DIR, "fine_tuned_model.pth")
    model_state_path = os.path.join(SAVE_DIR, "finetuned_model_state.pth")

    #if not general.basic_menu_model_option(model_path, ft_model):
    #    exit(0)

    if os.path.exists(model_state_path):
        print(f"[INFO] Found model state in {model_state_path}; Loading it...")
        ft_model.load_model(model_state_path)
        print("Model loaded.")
    else:

        print(f"[INFO] Model state not found in {model_state_path}; Training a new model...")

        # Fetches datasets, labels and create DataLoaders which will handle preprocessing images also.
        training_loader, valid_loader = general.get_dataloaders(
            return_study_id=False, pin_memory=is_cuda,
            use_bucket=True, verify_existence=False, full_data=True)

        # Train the model
        print("Starting training...")
        # NOTE: for other parameters, settings.py defines default values
        ft_model.train_model(training_loader)

        # Save the model
        print("Saving model...")
        ft_model.save_model(model_path)
        print(f"Model saved to {model_path}")

    # Any case: Evaluate the model

    print("Loading test dataset...")
    test_loader = general.get_test_dataloader(pin_memory=is_cuda,use_bucket=True,
                                              verify_existence=False, full_data=True)

    # Evaluate the model
    print("Starting evaluation...")
    metrics_dict = ft_model.model_evaluation(test_loader, save_stats=True, )
    print("Evaluation completed.")
    print("Metrics:", metrics_dict)

    # Save the metrics to a file
    metrics_file = os.path.join(SAVE_DIR, "fine_tuned_metrics.json")
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
