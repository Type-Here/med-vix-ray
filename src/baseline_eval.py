import torch, timm
import json
from PIL import Image

# Handler of datasets
import dataset.dataset_handle as dh
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms

from settings import MODELS_DIR, DATASET_PATH, DOWNLOADED_FILES, MIMIC_LABELS

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


def evaluate_predictions(predictions, test_dataset, threshold=0.5):
    """
    Valuta le predizioni fatte dal modello sul test set.
    Evaluate the predictions made by the src on the test set.

    :param predictions: Dictionary or Numpy Array with image paths as keys and predicted probabilities as values.
    :type predictions: dict or np.ndarray
    :param test_dataset: DataFrame with the true labels of the test set.
    :type test_dataset: pd.DataFrame
    :param threshold: Threshold to convert probabilities into binary labels.
    :type threshold: float

    :return: Dictionary with evaluation metrics.
    :rtype: dict
    """
    y_true = []
    y_pred = []

    # Check if predictions is a dictionary or numpy array
    if isinstance(predictions, np.ndarray):
        # If it's a numpy array, convert it to a dictionary with image paths as keys
        image_paths = dh.fetch_image_from_csv(test_dataset, DATASET_PATH, csv_kind='test')
        predictions = {image_path: predic for image_path, predic in zip(image_paths, predictions)}
    elif not isinstance(predictions, dict):
        raise ValueError("Predictions must be a dictionary or a numpy array.")

    # Iterate through the predictions
    for image_path, pred_probs in predictions.items():
        # Get dicom_id from the image path
        dicom_id = os.path.basename(image_path).split('.')[0]

        # Obtain the true labels from the dataset
        true_labels = test_dataset.loc[test_dataset['dicom_id'] == dicom_id, MIMIC_LABELS].values[0]

        # Check if the predicted probabilities are a tensor: if it is, convert it to numpy
        if isinstance(pred_probs, torch.Tensor):
            pred_probs = pred_probs.cpu().numpy()

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



def baseline_evaluation(testing_dataset):
    """
    Evaluate the baseline src to check its base performance, without any fine-tuning.
    This function loads the baseline src, processes the test dataset, and evaluates the predictions.
    :param testing_dataset: The test dataset to evaluate.
    :type testing_dataset: pd.DataFrame

    :return: Dictionary with predictions for each image. Keys are image paths, values are predicted probabilities.
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

    # Get all image paths from the test dataset
    image_paths = dh.fetch_image_from_csv(testing_dataset, DATASET_PATH, csv_kind='test') # TODO update parameters
    print("Image paths:", image_paths)

    # Define the image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization values standard for ImageNet
    ])

    # Obtain a prediction for each image
    predictions = {}
    num_images = len(image_paths)
    print("Number of images to evaluate:", num_images)
    print("Evaluating images...")
    print("Starting predictions...")

    count = 0
    # Iterate through the images and make predictions
    for image_path in image_paths:
        # Load the image
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
            # Preprocess the image (resize, normalize, etc.)
            image = preprocess(image)
            # Add batch dimension - PyTorch models expect a batch of images
            # Unsqueeze the image from [3, 256, 256] to [1, 3, 256, 256]
            image = image.unsqueeze(0)

            # Make a prediction
            with torch.no_grad():
                logits = model(image)
                probs = torch.sigmoid(logits).numpy().flatten()  # Converts logits in probabilities
                predictions[image_path] = probs
        if count % 100 == 0:
            print(f"Processed {count}/{num_images} images.")
        count += 1

    print("Finished predictions.")
    # Save predictions to a file
    return predictions


def load_test_set():
    """
    Load the test dataset. If it doesn't exist, create a new one.
    :return:
    test_dataset: The test dataset to evaluate.
    :rtype: pd.DataFrame
    """
    # Load the test dataset
    try:
        test_dataset = dh.load_ready_dataset(phase='test')
    except FileNotFoundError:
        print("Test dataset not found. Creating a new one.")
        merged_data = dh.dataset_handle(partial_list=DOWNLOADED_FILES)
        _, _, test_dataset = dh.split_dataset(merged_data)
    return test_dataset


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
    test_set = load_test_set()
    print("Test dataset loaded.")

    # Run the baseline evaluation
    print("Running baseline prediction...")
    pred = baseline_evaluation(test_set)

    print("Baseline prediction completed.")
    print("Number of predictions:", len(pred))

    # Save the predictions to a file
    pred_file = os.path.join(SAVE_DIR, 'baseline_predictions.json')
    save_predictions_to_json(pred, filename=pred_file)
    print(f"Predictions saved to {pred_file}.")

    # Evaluate the predictions
    print("Baseline prediction evaluation starting...")
    metrics_calculated = evaluate_predictions(pred, dh.load_ready_dataset(phase='test'))
    print("Evaluation metrics:", metrics_calculated)

    # Save the metrics to a file
    save_metrics_to_file(metrics_calculated, save_dir=SAVE_DIR)
    print("Metrics saved to file metrics.json.")