"""
File: region_encoder_graph_integration.py (updated)
-------------------------------------------------
Key changes 👇
1. **Pre‑computable sign embeddings**
   * `SignSemanticEmbedder` now supports *offline* embedding export & reload via
     `.save_precomputed()` / `.from_precomputed()`.
   * If you give a `precomp_path`, the heavy Sentence/BioBERT model is **not**
     loaded at run‑time – only a fixed `nn.Parameter` (learnable or frozen).
2. **Switchable fine‑tuning**
   * Pass `fine_tune=False` to keep the embedding vector learnable **without**
     gradient flowing through the large language model (so only a 128×N matrix
     is updated).
3. **Small cleanup** around device moving + dtype.

Usage example
-------------
```python
# ----- step‑0: offline (once) -----
embedder = SignSemanticEmbedder(
    sign_labels,
    device="cuda",
    encoder_name="microsoft/BiomedVLP-CXR-BERT-general",
)
embedder.save_precomputed("sign_init.pt")

# ----- training  (resource‑limited env) -----
embedder = SignSemanticEmbedder.from_precomputed(
    "sign_init.pt", learnable=True, device=t_device
)
# embedder(sign_ids) returns [B, 128]
```
-----------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from typing import List, Union, Optional

from settings import MANUAL_GRAPH

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # Will raise if we need it at run‑time


class SignSemanticEmbedder(nn.Module):
    """Embedder for radiological *sign* labels.

    Two working modes:
    1. **Online encode**  – load SBERT/BioBERT model and compute embeddings on
       the fly (optionally fine‑tune).
    2. **Pre‑computed**   – load a tensor saved with `save_precomputed`; only a
       small `proj` layer (or nothing) is learnable → tiny memory footprint.
    """

    def __init__(
        self,
        sign_labels: List[str],
        device: Union[str, torch.device] = "cpu",
        *,
        #encoder_name: str = "dmis-lab/biobert-base-cased-v1.1",
        encoder_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
        proj_dim: int = 128,
        fine_tune: bool = False,
        precomp_path: Optional[str] = None,
        learnable: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.proj_dim = proj_dim

        if precomp_path is not None:
            # ---- Lightweight path ----
            emb = torch.load(precomp_path, map_location=self.device)  # [N, D]
            if learnable:
                self.sign_emb = nn.Parameter(emb)        # learns small matrix
            else:
                self.register_buffer("sign_emb", emb)   # frozen
            self.encoder = None
        else:
            # ---- Full encoder path ----
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed")
            self.encoder = SentenceTransformer(encoder_name).to(self.device)
            self.encoder.eval()

            with torch.no_grad():
                emb = self.encoder.encode(sign_labels, convert_to_tensor=True)
            # Optionally detach & keep as param (tiny) instead of full model
            if fine_tune:
                self.sign_emb = nn.Parameter(emb)  # downstream fine‑tuning
            else:
                self.register_buffer("sign_emb", emb)
                # freeze encoder to save RAM
                for p in self.encoder.parameters():
                    p.requires_grad = False

        # Small projection to common dim used in RegionEncoder / Graph
        in_dim = emb.size(-1)
        self.proj = nn.Linear(in_dim, proj_dim)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def save_precomputed(self, out_path: str):
        """Export the *projected* sign embeddings → .pt file."""
        emb = self.forward(torch.arange(self.sign_emb.size(0), device=self.device))
        torch.save(emb.cpu(), out_path)

    @classmethod
    def from_precomputed(cls, path: str, *, learnable: bool = False, device="cpu"):
        emb = torch.load(path, map_location=device)  # [N, 128]
        dummy_labels = [f"sign_{i}" for i in range(emb.size(0))]

        # inizializza oggetto con proj_dim=emb.size(-1) così non crea encoder
        obj = cls(
            dummy_labels,
            device=device,
            proj_dim=emb.size(-1),  # =128
            precomp_path=None  # -> salta il download del modello
        )

        # rimpiazza la Linear con un’Identity (niente pesi)
        obj.proj = nn.Identity()

        if learnable:
            obj.sign_emb = nn.Parameter(emb.to(device))
        else:
            obj.register_buffer("sign_emb", emb.to(device))

        return obj

    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return projected embedding for indices *idx* (shape [B])."""
        base = self.sign_emb[idx]  # [B, D_in]
        return self.proj(base)     # [B, proj_dim]


if __name__ == "__main__":
    import json

    print("# ======== START SIGN EMBEDDING EXTRACTION ========")

    print("Loading sign labels and graph from JSON files...")
    # Load json graph:
    json_graph_path = MANUAL_GRAPH
    json_graph = json.load(open(json_graph_path, "r"))

    if json_graph is None:
        raise ValueError(f"Failed to load JSON graph from {json_graph_path}")
    print(f"Graph loaded with {len(json_graph)} nodes.")

    # Extract sign labels from the graph
    print("Generating Embedding for each sign...")
    labels = {}
    for node in json_graph["nodes"]:
        if node["type"] != "sign":
            continue
        text = node["label"] + ": " + node["description"]
        labels[node["id"]] = text

    print(f"Extracted {len(labels)} sign labels.")

    # Create the SignSemanticEmbedder instance
    print("Creating SignSemanticEmbedder...")
    list_labels = list(labels.values())
    embedder = SignSemanticEmbedder(list_labels, device="cpu", fine_tune=False)
    print(embedder(torch.tensor([0, 1, 2])))  # Get embeddings for the first 3 signs
    print("Embedding extraction complete.")

    # Save the precomputed embeddings to a file
    embedder.save_precomputed("signs_vect.pt")
    print("Precomputed embeddings saved to 'signs.pt'.")

    # Load the precomputed embeddings to verify
    print("Loading precomputed embeddings for verification...")
    loaded_embedder = SignSemanticEmbedder.from_precomputed("signs_vect.pt", learnable=True, device="cpu")
    print(loaded_embedder(torch.tensor([0, 1, 2])))  # Load and get embeddings again

    print("Extraction and saving complete.")
    print("You can now use the 'signs.pt' file in your models.")
    print("Exiting the script...")
