import concurrent.futures
from transformers import AutoTokenizer

import spacy
# Do not remove this import, it is used by spacy on adding the negex component to the pipeline
from negspacy.negation import Negex
import os, json
from settings import MIMIC_REPORT_DIR, NUM_WORKERS, RADLEX_DATA_DIR

#MODEL_NAME = "en_core_sci_md"
# sci_bert_url = https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
# bio_url = https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
def _load_model(model_name = "en_core_sci_scibert", ent_types=None):
    """
    Load the NER model and add the negex component to it.
    Args:
        model_name (str): Name of the NER model to load.
        ent_types (list): List of entity types to be used by negex.
    Returns:
       Language: Loaded the Spacy Model with NER and negex component.
    """
    if ent_types is None:
        ent_types = ["ENTITY"]

    nlp = spacy.load(model_name)
    nlp.add_pipe("negex", last=True, config={"ent_types": ent_types})
    # nlp.max_length = 3000000 # Set max length to avoid errors with long reports
    return nlp


def _save_dict_to_json(keywords_dict, path):
    """
    Save the keywords dictionary to a JSON file.
    Args:
        keywords_dict (dict): dictionary with keywords and their negex labels to save.
        path (str): path to save the JSON file.
    """
    with open(path, "w", encoding='utf-8') as file:
        json.dump(keywords_dict, file, indent=4)


def _run_list_of_reports(reports_path, nlp_model, num_workers=NUM_WORKERS):
    """
    Run NER on a list of reports. It uses a ThreadPoolExecutor to run
    the NER model on multiple reports in parallel.
    Args:
        reports_path (list): List of paths to the reports.
        nlp_model (spacy model): Pre-trained NER model.
        num_workers (int): Number of parallel workers for processing.
    Returns:
        dict: Dictionary with DICOM ID as key and a dictionary of entities
            and their negex labels as value.
    """
    keywords = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        tokenizer_sci = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        processed = 0
        total_reports = len(reports_path)
        results = executor.map(lambda path: _ner_single_report(path, nlp_model, tokenizer_sci), reports_path)
        for result in results:
            processed += 1
            progress = (processed / total_reports) * 100
            if processed % 100 == 0:
                print(f"Progress: {progress:.2f}%")
            keywords.update(result)
    return keywords


def _ner_single_report(path, nlp_model, tokenizer, max_length=512, stride=128):
    """
    Run NER on a single report. It extracts the text from the report and runs the NER model on it,
    obtaining entities as keys and negex labels as values in a dictionary.

    Args:
        path (str): Path to the report.
        nlp_model (spacy model): Pre-trained NER model with a .max_length attribute.
    Returns:
        dict: Dictionary with the dicom_id as key and a dictionary of entities
              and their negex labels as value.
    """
    # Read the file (the file will be closed automatically)
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Extract a dicom_id from the filename
    dicom_id = os.path.basename(path).split('.')[0]

    chunks = __sliding_window_chunks(text, tokenizer, max_length, stride)
    entities = {}
    for chunk in chunks:
        doc = nlp_model(chunk)
        for ent in doc.ents:
            if ent.text not in entities:
                entities[ent.text] = ent._.negex  # Ensure your spaCy model has this attribute set up.

    return {dicom_id: entities}


def __sliding_window_chunks(text, tokenizer, max_length, stride):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )
    input_ids_chunks = encoding["input_ids"]
    chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in input_ids_chunks]
    return chunks


def __generate_chunks(text, max_chunk_size):
    # Split on whitespace boundaries to avoid breaking words
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def _generator_paths_from_dir(path):
    """
    Walk through the directory and return all files.
    :param path: Path to the directory.
    :return: List of file paths.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                yield os.path.join(root, file)


if __name__ == "__main__":
    print("Starting...")
    print("Loading NER model...")
    nlp_ner = _load_model()

    print("Starting NER on MIMIC reports...")
    # Obtain all report paths from the directory
    reports_paths = list(_generator_paths_from_dir(MIMIC_REPORT_DIR))
    print("Found {} reports.".format(len(reports_paths)))

    # Run NER on the reports
    keywords_dictionary = _run_list_of_reports(reports_paths, nlp_ner, num_workers=2)
    print("NER completed.")

    # Save the keywords to a JSON file
    save_file = os.path.join(RADLEX_DATA_DIR, "keywords_ner.json")
    _save_dict_to_json(keywords_dictionary, save_file)
    print("NER Saved.")
    print("Exiting...")
