# import scispacy
import spacy
import os
import concurrent.futures
from collections import Counter

from settings import MIMIC_REPORT_DIR, NUM_WORKERS, RADLEX_DATA_DIR
# MODEL = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"

# Load the SciSpacy src
MODEL_NAME = "en_core_sci_md"
ner_model = spacy.load(MODEL_NAME)


def __extract_medical_entities(text):
    """
        Extract medical entities from MIMIC reports using SciSpacy.

        Args:
            text (str): Medical report text.

        Returns:
            list: list of extracted medical entities.
    """
    doc = ner_model(text)
    return [ent.text for ent in doc.ents]  # Return a list of entities


def __extract_text_from_mimic(mimic_report_folder):
    """
    Extract text from MIMIC reports.

    Args:
        mimic_report_folder (str): Path to the folder containing MIMIC reports.

    Returns:
        list: List of extracted text from MIMIC reports.
    """
    # Assuming the MIMIC reports are in text files
    report_texts = []
    for root, _, files in os.walk(mimic_report_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding="utf-8") as file:
                    report_texts.append(file.read())
    return report_texts


def __map_entities(entities):
    """
    From entities, extract the most relevant ones and count their occurrences.
    :param entities: List of extracted entities from MIMIC reports.
    :return: Dictionary with entity counts.
    """
    return dict(Counter(entities))  # Count occurrences of each entity, improving efficiency


def get_entities_from_reports(mimic_report_folder, threshold=0, num_workers=8):
    """
    Improve the graph using MIMIC reports.
    :param mimic_report_folder: Path to the folder containing MIMIC reports.
    :param threshold: Minimum count for entities to be included in the graph.
    :param num_workers: Number of parallel workers for processing.
    :return: Dictionary with entity counts.
    """
    print("📂 Estrazione testo dai report...")
    report_texts = __extract_text_from_mimic(mimic_report_folder)

    print("🧠 Avvio analisi NLP con SciSpacy in parallelo...")
    all_entities = []

    # Use ProcessPoolExecutor to parallelize entity extraction
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(__extract_medical_entities, report_texts)

    # Collect all entities from the results
    for entities in results:
        all_entities.extend(entities)

    print("📊 Mappatura delle entità e conteggio...")
    entity_map = __map_entities(all_entities)

    # Filter out entities with low frequency (if threshold > 0)
    if threshold > 0:
        entity_map = {k: v for k, v in entity_map.items() if v > threshold}

    # Ordering the entity map by frequency
    entity_map = dict(sorted(entity_map.items(), key=lambda item: item[1], reverse=True))

    # Print the top entities
    print("🔝 Entità più comuni:")
    for entity, count in list(entity_map.items())[:10]:  # Show only the top 10 entities
        print(f"{entity}: {count}")

    return entity_map


def save_entity_map_to_file(entity_map, output_file):
    """
    Save the entity map to a JSON file.
    :param entity_map: Dictionary with entity counts.
    :param output_file: Path to the output JSON file.
    """
    import json
    with open(output_file, 'w') as f:
        json.dump(entity_map, f, indent=4)

# ===== EXECUTION =====
if __name__ == "__main__":
    print("🚀 Avvio recupero entità da report MIMIC...")
    rel_entities = get_entities_from_reports(MIMIC_REPORT_DIR, num_workers=NUM_WORKERS)  # Use NUM_WORKERS from settings

    # Save the entity map to a JSON file
    save_output = os.path.join(RADLEX_DATA_DIR, "entity_map.json")
    save_entity_map_to_file(rel_entities, save_output)

    print("✅ Completato!")
