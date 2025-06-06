# Med-ViX-Ray

**Med-ViX-Ray** is a research framework developed for my Bachelor’s thesis in Computer Science at the University of Salerno, focused on explainable AI (XAI) and ontology-driven reasoning in medical imaging. The project aims to advance interpretability in deep learning for chest X-rays by integrating domain knowledge through structured graphs and specialized modules.

---

## Project Highlights

- **Explainable AI (XAI) for Medical Imaging:**  
  Med-ViX-Ray prioritizes explainability by injecting ontology-based knowledge (RadLex, manually curated graphs) directly into the attention mechanisms of Vision Transformers (ViT/Swin).  
  - **Ontology-Weighted Attention:** Medical concepts and their relationships are encoded as edges with dynamic weights, directly influencing the attention maps during training and inference.
  - **Sign Nudging Module:** Feature statistics from "sign" nodes in the knowledge graph are used to nudge model behavior, providing both regularization and interpretability.
  - **Entity-Report Weak Supervision:** NLP-extracted entities from clinical reports are used as weak supervision signals. These entities help infer and update edge weights as link probabilities, modeled with Beta distributions, and influence the learning process by inferring sign activations during the first epochs of training.
- **Graph-Based Knowledge Injection:**  
  The RadLex ontology and manually constructed graphs inform both the structure and the learning dynamics of the model, supporting a blend of data-driven and knowledge-driven inference.
- **Flexible Torch Model:**  
  The main model can be imported and used like any standard PyTorch model—just pass a chest X-ray image to obtain predictions and XAI outputs.

---

**Note:**  
Results and performance data will be added to this repository as soon as they become available following completion of training and evaluation.

---

## Model

The core model architecture is based on the Swin V2 Base transformer, adapted for multi-label classification using the 14 standard pathology labels from the MIMIC-CXR-JPG dataset.

Three model variants are provided in this repository:

- **Baseline:** Uses the pretrained Swin V2 Base model *as-is*, without modification.
- **Fine-Tuned:** The classifier head is adapted for the MIMIC label set, and the input layer and last two layers of the Swin model are fine-tuned.
- **Med-ViX-Ray:** Incorporates all customizations described above, including ontology-injected attention, sign nudging, and entity-driven weak supervision.

---

## Dataset

- **Training Data:**  
  Med-ViX-Ray is trained on the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) from PhysioNet.  
  **Note:** No data, model weights, or learned embeddings are distributed in this repository. See the Privacy and Data Policy section below.

---

## Installation

> **Requirements:**  
> - Python 3.8+  
> - PyTorch, torchvision, timm, scikit-learn, SciSpacy, SpaCy  
> - (Optional) AMD ROCm / CUDA for GPU acceleration

1. Clone the repository:
    ```bash
    git clone https://github.com/Type-Here/med-vix-ray.git
    cd med-vix-ray
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download and organize the MIMIC-CXR-JPG dataset as specified in `settings.py`.  
   You must apply for access and comply with all PhysioNet [usage policies for MIMIC data](https://physionet.org/about/).

---

## Usage

### Train or Evaluate the Model

```bash
python src/med-vix-ray.py
```
- The script will initialize the ontology graph, configure edge weights, set up the sign nudging module, and start training.
- Output includes: model checkpoints, architecture summaries, and training logs (saved in `src/models/`).

### Import and Use as a Torch Model

You can import the main model class and use it as a standard PyTorch module:
```python
from src.med_vix_ray import SwinMIMICGraphClassifier

model = SwinMIMICGraphClassifier()
output = model(<image_tensor>)
```

---

## Directory Structure

- `src/` — Model logic, training script, and utilities.
- `dataset/` — Dataset loading, preprocessing, and splitting.
- `ontology/` — Ontology management, entity extraction, graph construction.
- `medical/` — Clinical report processing and weak supervision modules.
- `settings.py` — All configuration and hyperparameters.

---

## Privacy and Data Policy

Med-ViX-Ray was developed and trained using the MIMIC-CXR-JPG dataset, which contains sensitive health information.  
**No data, pretrained weights, or any part of the learned model will be published or distributed** with this repository.  
This is to comply with the MIMIC dataset’s usage policy and to protect patient privacy, as research has shown that embeddings may sometimes leak information about training data.

**To use this framework, you must have your own authorized access to MIMIC-CXR-JPG and comply with all privacy and usage requirements.**

---

## Acknowledgements

- The [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the [PhysioNet](https://physionet.org/) community
- [RadLex Ontology](https://www.rsna.org/research/rsna-radlex) for medical knowledge graphs
- [SciSpacy](https://allenai.github.io/scispacy/) and the SpaCy team for NLP tools
- PyTorch, Timm, and the open-source ML ecosystem
- Inspiration and guidance from academic advisors, clinicians, and the open research community

---

## License

[MIT License](./LICENSE)

---

## No Warranties

This software is provided “as is,” without warranty of any kind. No guarantee is made regarding the correctness, performance, or fitness for any particular purpose. Use at your own risk.

---