import os, json
import spacy
import difflib
from settings import MANUAL_GRAPH, MANUAL_GRAPH_DIR


def _load_model():
    """
    Load the SciSpacy NER model.

    Returns:
        model: Loaded SciSpacy NER model.
    """
    model_name = "en_core_sci_scibert"
    model = spacy.load(model_name)
    return model


def _get_graph_findings_entities(graph_json, model):
    """
    Extract the findings entities from the graph file.

    Args:
        graph_json (dict): JSON graph data.

    Returns:
        dict: Dictionary of findings entities from the graph. Key: finding_id, Value: list of entities.
    """
    findings_entities = {}
    for node in graph_json["nodes"]:
        if node.get("type") == "finding":
            doc = model(node["label"] + "," + node["synonyms"])
            findings_entities[node["id"]] = [ent.text for ent in doc.ents]

    return findings_entities


def _compare_entities(report_entities, findings_entities):
    """
    Compare the entities extracted from the graph with the findings entities.

    Args:
        report_entities (dict): dict of entities extracted from the MIMIC reports. Key: dicom_id, Value: list of entities.
        findings_entities (dict): Dictionary of findings entities from the graph. Key: finding_id, Value: list of entities.

    Returns:
        dict: Dictionary with the comparison results; Keys are DICOM IDs and values are lists of finding IDs.
    """

    def token_cmp(find_tokens,repo_entity, dicom_repo_id, finding_tok_id):
        if dicom_repo_id not in comparison_results:
            comparison_results[dicom_repo_id] = []

        for token in find_tokens:
            if difflib.SequenceMatcher(None, repo_entity, token).ratio() > 0.8:
                comparison_results[dicom_repo_id].append(finding_tok_id)
                return

    comparison_results = {}
    for dicom_id, rep_entity in report_entities.items():
        for finding_id, tokens in findings_entities.items():
            # Use difflib to check for similarity
            token_cmp(tokens, rep_entity, dicom_id, finding_id)


    return comparison_results


def compare_entities_in_report(graph_json, reports_entities_dict):
    """
        Run the comparison of entities extracted from the reports
        with the findings entities of the graph for each report.
        Args:
            graph_json (dict): JSON graph data.
            reports_entities_dict (dict): Dictionary with DICOM ID as key and a dictionary of entities
                and their negex labels as value.
        Returns:
            list: List of dictionaries with the comparison results for each report.
    """
    matching_findings = []
    count = 0
    for repo in reports_entities_dict:
        matching_findings.append(_compare_entities(reports_entities_dict[repo], graph_json))
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} reports.")

    return matching_findings



if __name__ == "__main__":
    # Example usage
    if not os.path.exists(MANUAL_GRAPH):
        raise FileNotFoundError("Graph JSON file not found.")
    repo_entities_path = "path/to/reports_entities.json"

    if not os.path.exists(repo_entities_path):
        raise FileNotFoundError("Reports entities JSON file not found.")

    with open(repo_entities_path, 'r', encoding='utf-8') as file:
        reports_json = json.load(file)

    with open(MANUAL_GRAPH, 'r', encoding='utf-8') as file:
        graph_json = json.load(file)

    model = _load_model()
    findings_entities = _get_graph_findings_entities(graph_json, model)

    comparison_results = compare_entities_in_report(findings_entities, reports_json)
    print("First 5 elements: ", comparison_results[:5])

    # Save the comparison results to a JSON file
    output_path = MANUAL_GRAPH_DIR + "comparison_results.json"
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(comparison_results, file, indent=4)