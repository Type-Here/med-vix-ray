import json
import os
import random
from pathlib import Path
import numpy as np
import torch

from settings import MODELS_DIR, MANUAL_GRAPH
from src import general
from src.eval.eval import evaluate_multilabel
from src.fine_tuned_model import SwinMIMICClassifier
from src.med_vix_ray import SwinMIMICGraphClassifier
from bootstrap import bootstrap_multilabel
from metrics import validate_multilabel_inputs

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
        args.out = f"results/test_eval_{args.n_boot}boot_{args.alpha}alpha_finetuned.json"

    print("Starting Fine Tuned Model Eval...")

    # Check for device
    print("Checking for device...")
    t_device = torch.device("cpu" if torch.version.hip else
                            ("cuda" if torch.cuda.is_available() else "cpu"))
    is_cuda = torch.cuda.is_available() and not torch.version.hip
    # ROCm is not supported yet
    print(f"Using device: {t_device}")

    ft_model = SwinMIMICClassifier(device=t_device).to(t_device)

    print(" -- Verifying save directory and loading model state if exists --")
    SAVE_DIR = os.path.join(MODELS_DIR, "fine_tuned")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # If it still doesn't exist exit, so avoid errors after waiting all training
    if not os.path.exists(SAVE_DIR):
        print("Unable to create save dir. Exiting...")
        exit(1)

    # Load Model if exists
    #model_path = os.path.join(SAVE_DIR, "med_vixray_model.pth")
    model_state_path = os.path.join(SAVE_DIR, "finetuned_model_state.pth")

    # if not general.basic_menu_model_option(model_path, ft_model):
    #    exit(0)

    if os.path.exists(model_state_path):
        print(f"[INFO] Found model state in {model_state_path}; Loading it...")
        ft_model.load_model(model_state_path)
        print("Model loaded.")
    else:

        print("[INFO] Model State not found, cannot load model. Exiting...")
        exit(1)


    if not args.npz:
        print("No npz file provided. Using dataloader...")
        test_loader = general.get_test_dataloader(pin_memory=True, full_data=True, use_bucket=False)

        # Set the worker_init_fn to ensure reproducibility in each worker# Create a CPU Generator seeded for reproducibility and attach it to the DataLoader
        generator = torch.Generator(device='cpu')
        generator.manual_seed(args.seed)
        test_loader.generator = generator
        # Ensure per-worker seeding still uses the provided seed
        test_loader.worker_init_fn = lambda worker_id: seed_worker(worker_id)


        out = evaluate_multilabel(ft_model, test_loader=test_loader, n_boot=args.n_boot,
                                      seed=args.seed, n_bins=args.n_bins)

        #Y_true, Y_prob, label_names = out['Y_true'], out['Y_prob'], out.get('label_names')

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
