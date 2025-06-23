from evaluate import load
from datasets import load_dataset

from src.llm.inference import generate_report

bleu = load("bleu")
bertscore = load("bertscore")

ds = load_dataset("json", data_files="data/val.jsonl")["train"]
preds, refs = [], []
for r in ds:
    preds.append(generate_report(r["prompt_json"]))  # wrap your gen
    refs.append(r["text"])

print("BLEU:", bleu.compute(predictions=preds, references=[[t] for t in refs])["bleu"])
print("BERTScore:", bertscore.compute(predictions=preds, references=refs)["f1"][0])
