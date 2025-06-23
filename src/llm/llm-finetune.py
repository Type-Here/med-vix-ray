from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch, bitsandbytes as bnb

"""
This module, `llm-finetune.py`, is designed for fine-tuning a pre-trained 
language model (LLM) using the LoRA (Low-Rank Adaptation) technique. 
The fine-tuned model is optimized for generating concise impressions 
from radiology reports, specifically in the context of thoracic imaging.

Key Features:
1. **Model Selection and Quantization**:
   - The script uses a pre-trained model, such as `microsoft/biogpt-large`, 
     which is loaded in 4-bit precision for efficient memory usage and faster training.
   - Quantization is configured using the `bitsandbytes` library to enable 
     low-bit computations while maintaining performance.

2. **LoRA Fine-Tuning**:
   - LoRA is applied to specific components of the transformer architecture 
     (e.g., `q_proj` and `v_proj` layers) to focus fine-tuning on the attention mechanism.
   - This approach reduces the number of trainable parameters, making the 
     fine-tuning process more efficient and cost-effective.

3. **Dataset Preparation**:
   - The training and validation datasets are loaded from JSONL files, 
     which contain pairs of prompts and target texts.
   - The prompts are structured to simulate a radiologist's task, 
     while the target texts represent the expected impressions.
   - Tokenization is performed to prepare the data for training, 
     with input and target sequences padded and truncated to fixed lengths.

4. **Training Configuration**:
   - The script uses the `Trainer` class from the `transformers` library 
     to manage the training process.
   - Key training parameters, such as batch size, gradient accumulation, 
     learning rate, and number of epochs, are configurable.
   - Mixed precision training (`fp16`) is enabled for faster computations.

5. **Model Saving**:
   - After training, the fine-tuned model and tokenizer are saved to a 
     specified directory for later use in inference tasks.

Usage:
- This script is intended for developers and researchers working on 
  domain-specific language model fine-tuning, particularly in the medical field.
- The fine-tuned model can be used for generating concise and accurate 
  impressions from radiology reports, aiding in clinical decision-making.

Dependencies:
- `transformers`: For model loading, tokenization, and training.
- `peft`: For applying LoRA to the model.
- `datasets`: For loading and processing the training/validation datasets.
- `bitsandbytes`: For 4-bit quantization and efficient computations.
- `torch`: For PyTorch-based model training.

Note:
- Ensure that the required datasets (`train.jsonl` and `val.jsonl`) are 
  prepared and available in the `data` directory before running this script.
- The script is designed to work with GPU acceleration. Ensure that a 
  compatible GPU and CUDA environment are available for optimal performance.
"""

MODEL_NAME = "microsoft/biogpt-large"
# MODEL_NAME = "QizhiPei/biot5-plus-base"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
BATCH = 2
GRAD_ACC = 8
EPOCHS = 2
LR = 2e-4

# 1. tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token  # BioGPT has no pad token

# 2. model 4‑bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    quantization_config=bnb.nn.Linear4bit.config(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ),
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA adapters
"""
q_proj and v_proj are specific components of the transformer architecture:

These refer to the query projection (q_proj) and value projection (v_proj) layers 
in the attention mechanism of transformer models.

In self-attention, the input is projected into query (Q), key (K), and value (V) vectors. 
The q_proj and v_proj layers are linear transformations responsible 
for creating the query and value vectors, respectively.

Applying LoRA to these layers focuses fine-tuning on the attention mechanism, 
which is critical for model performance.
"""
peft_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# 4. dataset
# Load the dataset from JSONL files for training and validation
ds = load_dataset("json", data_files={"train": "data/train.jsonl",
                                      "val": "data/val.jsonl"})

# Define a function to tokenize the dataset
def tokenise(batch):
    # Tokenize the "prompt" field, truncating to 512 tokens and padding to the maximum length
    out = tok(batch["prompt"], truncation=True, max_length=512, padding="max_length")
    # Tokenize the "text" field (target), truncating to 256 tokens and padding to the maximum length
    with tok.as_target_tokenizer():
        tgt = tok(batch["text"], truncation=True, max_length=256, padding="max_length")
    # Add the tokenized target input IDs as labels for the model
    out["labels"] = tgt["input_ids"]
    return out

# Apply the tokenization function to the dataset, process in batches, and remove original columns
ds = ds.map(tokenise, batched=True, remove_columns=ds["train"].column_names)

# 5. trainer
args = TrainingArguments(
    output_dir="biogpt_lora",
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=args,
                  train_dataset=ds["train"], eval_dataset=ds["val"])
trainer.train()
# 6. save
model.save_pretrained("biogpt_lora/final")
tok.save_pretrained("biogpt_lora/final")
