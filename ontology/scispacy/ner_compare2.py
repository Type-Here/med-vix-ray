import json
import os.path

import pandas as pd
import difflib

from settings import MANUAL_GRAPH, MIMIC_REPORT_DIR

def _load_sign_nodes(path):
    """
        Method to load the sign nodes from the graph.
        Args:
            path (str): Path to the graph JSON file.
        Returns:
            dict: Dictionary containing the sign nodes and their attributes.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    sign_nodes = {}
    for node in data['nodes']:
        if node['type'] == 'sign':
            sign_nodes[node['id']] = {
                'label': node['label'],
                'synonyms': node.get('synonyms', []),
            }
    return sign_nodes

def _load_ner_reports(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def _load_csv_metadata(path):
    """
        Method to load the CSV metadata file.
        Args:
            path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: DataFrame containing the metadata.
    """
    return pd.read_csv(path)

def _calculate_similarity(entity, label, synonyms):
    """
        Method to calculate the similarity score between the entity and the label/synonyms using difflib.
        Args:
            entity (str): The entity extracted from the report.
            label (str): The label of the sign node.
            synonyms (list): List of synonyms for the sign node.
        Returns:
            float: Similarity score between the entity and the label/synonyms.
    """
    similarity = difflib.SequenceMatcher(None, entity, label).ratio()
    for synonym in synonyms:
        sim = difflib.SequenceMatcher(None, entity, synonym).ratio()
        if sim > similarity:
            similarity = sim
    return similarity


# Return dict structure expected:
# { study_id:
#   { "sign_node_id": [s_gt, polarity],
#       ...,
#     "dicom_ids": [dicom_id1, dicom_id2, ...],
#   }, ...
# }
# sign_node_id: id of the sign node in the graph
# s_gt: similarity score between report word and sign node label (or synonym)
# polarity: negex value (true/false) (inverted: true = present, false = not present)
def compare_signs_reports(f_graph_path, f_ner_path, f_metadata_path) -> dict:
    """
        Compare the signs in the reports with the findings in the graph.
        Args:
            f_graph_path (str): Path to the graph JSON file.
            f_ner_path (str): Path to the NER reports JSON file.
            f_metadata_path (str): Path to the CSV metadata file.
        Returns:
            dict: Dictionary containing the comparison results.
    """
    # Load the reports and metadata
    # Reports are in structured as:
    # { study_id:
    #   { "ner": negex_value, ... }
    # }
    # ner is an entity extracted from the report using NER scispacy.
    # Negex values are:
    #   - true: Not present
    #   - false: Present

    # Load sign nodes from the graph
    sign_nodes = _load_sign_nodes(f_graph_path)

    # Load the NER reports
    reports = _load_ner_reports(f_ner_path)

    # Metadata contains the dicom_id and the study_id, with multiple dicom_ids per study_id.
    metadata = _load_csv_metadata(f_metadata_path)

    # Initialize an empty dictionary to store the comparison results
    comparison_results = {}

    # Iterate through each study ID in the reports
    for study_id, ner_dict in reports.items():
        study = {'dicom_ids': metadata[metadata['study_id'] == study_id]['dicom_id'].tolist()}
        # Get the dicom IDs for the current study ID
        # Iterate through each entity in the NER dictionary
        for entity, negex_value in ner_dict.items():
            for sign_node_id, sign_node in sign_nodes.items():
                # Check if the entity matches the sign node label or synonyms
                similarity = _calculate_similarity(entity, sign_node['label'], sign_node['synonyms'])

                if similarity > 0.8:  # TODO Adjust the threshold as needed
                    # Save similarity score and inverted negex value
                    study[sign_node_id] = [similarity, not negex_value]

        comparison_results[study_id] = study

    return comparison_results



if __name__ == "__main__":

    # Paths of needed files
    print("Loading data...")
    graph_path = MANUAL_GRAPH
    ner_path = os.path.join(MIMIC_REPORT_DIR, "ner_reports", "save", "keywords_ner_reports_p10.npy.json")
    metadata_path = os.path.join(MIMIC_REPORT_DIR, "info", "metadata.csv")

    print("Comparing signs and reports...")
    compare_results = compare_signs_reports(graph_path, ner_path, metadata_path)

    print("Comparison completed.")
    print("Results:")
    # Print the comparison results
    # Print only the first 10 results for brevity
    for study_id_key, result in compare_results.items()[:10]:
        print(f"Study ID: {study_id_key}")
        print(result)

    print("Saving results...")
    #Save the results to a JSON file
    with open('comparison_results.json', 'w') as file:
        json.dump(compare_results, file, indent=4)

    print("Results saved to comparison_results.json")
    print("Done.")