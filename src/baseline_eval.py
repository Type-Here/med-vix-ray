import torch, timm
import json

# Handler of datasets
import os
import numpy as np
import general
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from settings import MODELS_DIR

# Create save src directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_metrics_to_file(metrics, output_file = "metrics.json", save_dir = MODELS_DIR):
    """
    Salva le metriche in un file JSON.
    Save the metrics to a JSON file.

    :param metrics: Dictionary with evaluation metrics.
    :type metrics: dict
    :param output_file: Path to the output JSON file.
    :type output_file: str
    :param save_dir: Directory where the output file will be saved.
    :type save_dir: str
    """
    save = os.path.join(save_dir, output_file)
    with open(save, 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_predictions(predictions, threshold=0.5):
    """
    Valuta le predizioni fatte dal modello sul test set.
    Evaluate the predictions made by the src on the test set.

    """
    y_true = []
    y_pred = []

    # Iterate through the predictions
    for _, entry in predictions.items():
        pred_probs = np.array(entry['probs'])
        true_labels = np.array(entry['labels'])

        # Convert the predictions to binary labels (0 or 1) based on the threshold
        pred_labels = (pred_probs >= threshold).astype(int)

        # Append the true and predicted labels to the lists
        y_true.append(true_labels)
        y_pred.append(pred_labels)

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if the shapes of y_true and y_pred match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")

    # Calculate multilabel metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    return metrics


def save_predictions_to_json(predictions, filename="predictions.json"):
    """
    Save predictions to a JSON file.

    :param predictions: Dictionary with image paths as keys and prediction arrays as values.
    :param filename: Name of the output JSON file.
    """
    if not predictions:
        print("❌ Errore: Nessuna predizione da salvare!")
        return

    json_ready_predictions = {}

    for img_path, preds in predictions.items():
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()  # Converts tensor in numpy array
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()  # Converts numpy array in list

        json_ready_predictions[img_path] = preds  # Adds in dictionary

    # Convert numpy arrays to lists for JSON serialization
    #json_ready_predictions = {img_path: preds.tolist() for img_path, preds in predictions.items()}

    preview = list(json_ready_predictions.items())[:3]
    print(" Preview of the first 3 predictions:\n", json.dumps(dict(preview), indent=4))

    # Save the predictions to a JSON file
    with open(filename, "w") as f:
        json.dump(json_ready_predictions, f, indent=4)

    print(f"Predictions saved successfully to {filename}")

def __baseline_simple_test():

    num_classes = 14

    # Adapts the src to the number of classes in MIMIC-CXR (14 classes)
    model = timm.create_model("swinv2_base_window8_256.ms_in1k", pretrained=True, num_classes=num_classes)
    print("📌 Original Model Head:", model.head)

    # Set the src to evaluation mode
    model.eval()
    # Define the image preprocessing pipeline
    test_image = torch.randn(1, 3, 256, 256)  # Simulate an input image
    output = model(test_image)

    print("📌 Output shape:", output.shape)  # Should Print: torch.Size([1, 14])


def baseline_evaluation(test_loader):
    """
    Evaluate the baseline src to check its base performance, without any fine-tuning.
    This function loads the baseline src, processes the test dataset, and evaluates the predictions.
    Args:
        test_loader(DataLoader) : The test dataset to evaluate.
    Returns:
        dict: Dictionary with predictions for each image. Keys are indices of the test set,
                values are dictionaries with probabilities and labels.
    """
    # Load the baseline src.
    # src = torch.load(os.path.join(MODELS_DIR, 'baseline_model.pth'))

    # Load Swin Transformer src
    model = timm.create_model("swinv2_base_window8_256.ms_in1k", pretrained=True, num_classes=14)

    # Adapts the src to the number of classes in MIMIC-CXR (with 14 classes)
    #src.head = torch.nn.Linear(src.head.in_features, num_classes)

    # Check if the src head is correctly modified
    print("✅ Model head after modification:", model.head)

    # Set the src to evaluation mode
    model.eval()

    # Obtain a prediction for each image
    predictions = {}
    num_images = len(test_loader)
    print("Number of images (batches) to evaluate:", num_images)
    print("Evaluating images...")
    print("Starting predictions...")

    count = 0
    # Iterate through the images and make predictions
    for image, labels in test_loader:
        # Make a prediction
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits).numpy().flatten()  # Converts logits in probabilities
            predictions[count] = {'probs' : probs, 'labels' : labels.numpy().flatten()}
        if count % 100 == 0:
            print(f"Processed {count}/{num_images} images.")
        count += 1

    print("Finished predictions.")
    # Save predictions to a file
    return predictions


if __name__ == "__main__":
    SAVE_DIR = os.path.join(MODELS_DIR, 'baseline')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting baseline evaluation...")
    simple_test = False

    if simple_test:
        __baseline_simple_test()
        exit(0)

    # Load the test dataset
    test_dataloader = general.get_test_dataloader(full_data=False, verify_existence=True,
                                             use_bucket=False, channels_mode="RGB")
    print("Test dataset loaded.")

    # Run the baseline evaluation
    print("Running baseline prediction...")
    pred = baseline_evaluation(test_dataloader)

    print("Baseline prediction completed.")
    print("Number of predictions:", len(pred))

    # Save the predictions to a file
    pred_file = os.path.join(SAVE_DIR, 'baseline_predictions.json')
    save_predictions_to_json(pred, filename=pred_file)
    print(f"Predictions saved to {pred_file}.")

    # Evaluate the predictions
    print("Baseline prediction evaluation starting...")
    metrics_calculated = evaluate_predictions(pred)
    print("Evaluation metrics:", metrics_calculated)

    # Save the metrics to a file
    save_metrics_to_file(metrics_calculated, save_dir=SAVE_DIR)
    print("Metrics saved to file metrics.json.")