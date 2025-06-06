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
    import concurrent.futures
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

    def process_study(args):
        rep_study_id, ner_dict = args
        # Extract the study_id from the report
        # report_study_id is in the format "sXXXX" so we remove the "s"
        study = {'dicom_ids': metadata[metadata['study_id'] == int(rep_study_id[1:])]['dicom_id'].tolist()}
        # Iterate through each entity in the NER dictionary
        for entity, negex_value in ner_dict.items():
            for sign_node_id, sign_node in sign_nodes.items():
                # Check if the entity matches the sign node label or synonyms
                similarity = _calculate_similarity(entity, sign_node['label'], sign_node['synonyms'])
                if similarity > 0.7:  # TODO Adjust the threshold as needed
                    # Save similarity score and inverted negex value
                    study[sign_node_id] = [round(similarity, 4), not negex_value]
        return rep_study_id, study

    comparison_results = {}
    total_items = len(reports)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_item = {executor.submit(process_study, item): item for item in reports.items()}
        for future in concurrent.futures.as_completed(future_to_item):
            study_id, study_dict = future.result()
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed / total_items * 100:.2f}% complete")
            comparison_results[study_id] = study_dict

    return comparison_results


def print_first_10_res(compare_result):
    print("Results:")
    # Print the comparison results
    # Print only the first 10 results for brevity
    k = 0

    for study_id_key, result in compare_result.items():
        print(f"Study ID: {study_id_key}")
        print(result)
        k += 1
        if k >= 10:
            break


if __name__ == "__main__":

    # Paths of needed files
    print("Loading data...")
    graph_path = MANUAL_GRAPH
    ner_dir = os.path.join(MIMIC_REPORT_DIR, "ner_reports", "save")
    metadata_path = os.path.join(MIMIC_REPORT_DIR, "info", "metadata.csv")

    print("Comparing signs and reports...")

    all_results = {}

    for ner in os.listdir(ner_dir):
        if ner.endswith(".json"):
            print(f" -- Comparing Report {ner}... -- ")
            ner_path = os.path.join(ner_dir, ner)  # Construct full path
            compare_results = compare_signs_reports(graph_path, ner_path, metadata_path)
        else:
            continue

        print(f"Comparison for {ner} completed.")
        print("Saving results...")
        # Save the results to a JSON file

        # Extract the number from the filename
        file_number = ner.split("_")[-1].split(".")[0]

        # Save the results to a JSON file
        with open(f'comparison_results_{file_number}.json', 'w') as file:
            json.dump(compare_results, file, indent=4)

        print(f"Results saved to comparison_results_{file_number}.json")

        # Adding the results to all_results
        all_results.update(compare_results)

    # Create a json file with all the results
    print("Saving file with all results grouped...")
    with open('all_comparison_results.json', 'w') as file:
        json.dump(all_results, file, indent=4)
    print("All results saved to all_comparison_results.json")

    print("Done. Exiting...")