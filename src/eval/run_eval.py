import json
import os
import random
from pathlib import Path
import numpy as np
import torch

from env_test.rocm_test import result
from settings import MODELS_DIR, MANUAL_GRAPH
from src import general
from src.eval.eval import evaluate_multilabel
from src.med_vix_ray import SwinMIMICGraphClassifier
from .bootstrap import bootstrap_multilabel
from .metrics import validate_multilabel_inputs

# Function to set the seed for each worker
def seed_worker(worker_id):
    # Derive a per-worker seed from the DataLoader / torch initial seed.
    # DataLoader uses the provided `generator` (seeded in the main process)
    # to set each worker's initial seed; `torch.initial_seed()` returns that value.
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    Y_true = data["Y_true"]
    Y_prob = data["Y_prob"]
    label_names = None
    if "label_names" in data:
        label_names = list(data["label_names"].tolist())
    return Y_true, Y_prob, label_names


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=False, help="NPZ with Y_true, Y_prob, optional label_names")
    p.add_argument("--out", required=False, help="Output JSON path")
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_bins", type=int, default=15)
    args = p.parse_args()

    if not args.out:
        args.out = f"results/test_eval_{args.n_boot}boot_{args.alpha}alpha.json"

    print("Starting Med-ViX-Ray model main...")
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

    print("Testing attention hook...")
    general.test_attention_hook(med_model.swin_model, t_device)
    print("[INFO] Overwritten attention forward function with hook-enabled version.")

    print(" -- Verifying save directory and loading model state if exists --")
    SAVE_DIR = os.path.join(MODELS_DIR, "med-vix-ray")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # If it still doesn't exist exit, so avoid errors after waiting all training
    if not os.path.exists(SAVE_DIR):
        print("Unable to create save dir. Exiting...")
        exit(1)

    # Load Model if exists
    #model_path = os.path.join(SAVE_DIR, "med_vixray_model.pth")
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
        print("[INFO] Model State not found, cannot load model. Exiting...")
        exit(1)


    if not args.npz:
        print("No npz file provided. Using dataloader...")
        test_loader = general.get_test_dataloader(pin_memory=True, full_data=True)

        # Set the worker_init_fn to ensure reproducibility in each worker# Create a CPU Generator seeded for reproducibility and attach it to the DataLoader
        generator = torch.Generator(device='cpu')
        generator.manual_seed(args.seed)
        test_loader.generator = generator
        # Ensure per-worker seeding still uses the provided seed
        test_loader.worker_init_fn = lambda worker_id: seed_worker(worker_id)


        out = evaluate_multilabel(med_model, test_loader=test_loader, n_boot=args.n_boot,
                                      seed=args.seed, n_bins=args.n_bins)
    else:
        Y_true, Y_prob, label_names = load_npz(args.npz)
        Y_true, Y_prob, label_names = validate_multilabel_inputs(Y_true, Y_prob, label_names)

        out = bootstrap_multilabel(
            Y_true,
            Y_prob,
            label_names=label_names,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
            n_bins=args.n_bins,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
