import json, torch


def generate_report(model, tokenizer, v_json_path:str):
    """
    Generate a report using the model and tokenizer.
    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for encoding the input.
        v_json_path: The path to the JSON file containing the input data.
    Returns:
        str: The generated report.
    """
    j = json.loads(open(v_json_path).read())

    prompt = (
      "### System:\nYou are an expert thoracic radiologist.\n\n"
      "### Input:\n"+json.dumps(j)+"\n\n"
      "### Task:\nWrite a concise but precise impression."
    )

    inp = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.cuda.amp.autocast():
        out = model.generate(**inp, max_new_tokens=120,
                             temperature=0.4, top_p=0.9)

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### Task:")[-1].strip()
