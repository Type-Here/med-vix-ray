import json, random, pathlib, pandas as pd
from tqdm import tqdm

# 1. vision_outputs_dir contains one JSON per study from your vision‑graph model.
vision_dir = pathlib.Path("vision_json")
# 2. MIMIC csv with impression column.
df = pd.read_csv("mimic_cxr_metadata.csv")
id2text = {row.study_id: row.impression for _, row in df.iterrows()}

records = []
for vp in tqdm(list(vision_dir.glob("*.json"))):
    j = json.loads(vp.read_text())
    sid = j["study_id"]
    if sid not in id2text:
        continue
    prompt = (
        "### System:\n"
        "You are an expert thoracic radiologist.\n\n"
        "### Input:\n"
        f"{json.dumps(j, ensure_ascii=False)}\n\n"
        "### Task:\n"
        "Write a concise impression."
    )
    records.append({"prompt": prompt, "text": id2text[sid]})

random.shuffle(records)
split = int(len(records)*0.8)
pathlib.Path("data").mkdir(exist_ok=True)
# Save the records to JSONL files
# train.jsonl for training
with open("data/train.jsonl","w") as f:
    for r in records[:split]:
        f.write(json.dumps(r)+"\n")
print("Train pairs:", split)

# Re-split the records for validation and test
records = records[split:]
split = int(len(records)*0.5)

# val.jsonl for validation
with open("data/val.jsonl","w") as f:
    for r in records[split:]:
        f.write(json.dumps(r)+"\n")
print("Validation pairs:", split)

# test.jsonl for testing
with open("data/test.jsonl","w") as f:
    for r in records[:split]:
        f.write(json.dumps(r)+"\n")
print("Test pairs:", split)