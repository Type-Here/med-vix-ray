import os, json
import torch
from settings import MODELS_DIR, MANUAL_GRAPH
import src.general as general
from src.med_vix_ray import SwinMIMICGraphClassifier

if __name__ == "__main__":
    """
       Ablation Study for Med-ViX-Ray model.
    """
    print("-" * 30)
    print("Starting Ablation Study for Med-ViX-Ray model...")
    print("-" * 30)


    print("Loading Graph JSON...")
    # Load the graph JSON file
    with open(MANUAL_GRAPH, 'r') as file:
        data_graph_json = json.load(file)

    # Check for device
    print("Checking for device...")
    t_device = torch.device("cpu" if torch.version.hip else
                            ("cuda" if torch.cuda.is_available() else "cpu"))
    is_cuda = torch.cuda.is_available() and not torch.version.hip
    # ROCm is not supported yet
    print(f"Using device: {t_device}")

    # Init Model
    med_model = SwinMIMICGraphClassifier(graph_json=data_graph_json, device=t_device).to(t_device)
    print("Model initialized.")
    # You can now train the model using the train_model method.
    # Example: model.train_model(train_loader)
    # Note: train_loader should be defined with your training dataset.


    print(" -- Verifying save directory and loading model state if exists --")
    SAVE_DIR = os.path.join(MODELS_DIR, "med-vix-ray")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # If it still doesn't exist exit, so avoid errors after waiting all training
    if not os.path.exists(SAVE_DIR):
        print("Unable to create save dir. Exiting...")
        exit(1)

    # Load Model if exists
    model_path = os.path.join(SAVE_DIR, "med_vixray_model.pth")
    model_state_path = os.path.join(SAVE_DIR, "med_vixray_model_state.pth")
    json_graph_path = os.path.join(SAVE_DIR, "med_vixray_model_graph.json")

    if os.path.exists(model_state_path):
        print("[INFO] Model State found! Trying to load it...")
        try:
            with open(json_graph_path, 'r') as file:
                data_graph_json = json.load(file)
        except (FileNotFoundError, OSError):
            print("[WARNING] Graph JSON file not found. Using default graph.")
            exit(1)

        med_model.load_model_from_state(state_dict_path=model_state_path, graph_json=data_graph_json)
        print("[INFO] Model State loaded!")
    else:
        raise FileNotFoundError("Model state file not found. Please train the model before evaluation.")

    # Ablation: Do not use nudging
    med_model.is_using_nudger=False
    print("Ablation: Nudging disabled.")

    med_model.eval()

    # Evaluate the model
    print(" -- Starting evaluation --")
    # Load the test dataset
    print("Loading test dataset...")
    test_loader = general.get_test_dataloader(full_data=True, pin_memory=is_cuda,
                                              use_bucket=False, verify_existence=False,
                                              channels_mode="RGB")
    print("Test dataset loaded.")

    metrics_dict = med_model.model_evaluation(test_loader, save_stats=False)
    print("Evaluation completed.")
    print("Metrics:", metrics_dict)

    # Save the metrics to a file
    metrics_file = os.path.join(SAVE_DIR, "med-vix_metrics_ablation_nudger.json")
    with open(metrics_file, 'w') as file:
        file.write(str(metrics_dict))
    print(f"Model Metrics saved to {metrics_file}")

    print("Ablation study completed.")
    print("Exiting...")
    print("-" * 30)
