"""
    Main for Inference with Med-ViX-Ray model.
"""
import os
import json
import torch
from PIL import Image

from src.med_vix_ray import SwinMIMICGraphClassifier
from settings import MANUAL_GRAPH, MODELS_DIR
from src.preprocess import preprocess_image

# Image path to predict
IMAGE_PATH = "image.png"
# View position for the image AP or PA
VIEW_POSITION = "AP"

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
        # Set the model to evaluation mode
        med_model.eval()
    else:
        print("Unable to find model state... Exiting!")
        exit(1)

    mimic_labels = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
                    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
                    "Lung Opacity","No Finding","Pleural Effusion","Pleural Other",
                    "Pneumonia","Pneumothorax","Support Devices"]


    print("Model loaded successfully. Ready for inference.")

    # Preprocess the image
    image = Image.open(IMAGE_PATH).convert("RGB")
    image = preprocess_image(image, channels_mode="RGB", view_position=VIEW_POSITION, augment=False, is_train=False)
    image = image.unsqueeze(0).to(t_device)  # Add batch dimension

    with torch.no_grad():
        # Forward pass through the model
        logits = med_model(image)

        attention_map = med_model.att_maps_batch.detach() if med_model.att_maps_batch is not None else None
        feat_map = med_model.features_dict_batch
        signs_found = med_model.signs_found

        # Convert logits to probabilities
        pred_probs = torch.sigmoid(logits).cpu().numpy()

    pred_labels = {label: prob for label, prob in zip(mimic_labels, pred_probs.flatten())}
    print(" -- Raw Predictions: -- ")
    for label, prob in pred_labels.items():
        print(f"{label}: {prob:.4f}")

    print("\n -- Predictions with Threshold 0.5: -- ")
    # Apply threshold to predictions
    threshold = 0.5
    for label, prob in pred_labels.items():
        if prob >= threshold:
            print(f"{label} Predicted")

    # Generate plot for attention map (save it)
    if attention_map is not None:
        import matplotlib.pyplot as plt
        plt.imshow(attention_map.cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.title("Attention Map")
        plt.colorbar()
        plt.savefig(os.path.join(SAVE_DIR, "attention_map.png"))
        plt.close()

    #Generate plot for feature map (save it)
    if feat_map is not None:
        plt.imshow(feat_map.cpu().numpy(), cmap='gray', interpolation='nearest')
        plt.title("Feature Map")
        plt.colorbar()
        plt.savefig(os.path.join(SAVE_DIR, "feature_map.png"))
        plt.close()

    # Print signs found
    if signs_found is not None:
        print("-- Signs found:")
        for sign in signs_found:
            print(f"- {sign}")



