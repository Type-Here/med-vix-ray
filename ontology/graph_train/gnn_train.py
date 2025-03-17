import spacy
from gensim.models import KeyedVectors
from difflib import get_close_matches
from settings import SIMILARITY_THRESHOLD, POSITIVE_WEIGHT_CORR, NEGATIVE_WEIGHT_CORR, POSITIVE_WEIGHT_FINDING, \
    NEGATIVE_WEIGHT_FINDING

# ========================================================== NODES VALUES ATTRIBUTION =========================================================

# 1. Load the pre-trained SciSpacy src for the medical domain
nlp = spacy.load("en_core_sci_sm")

# 2. Load a pre-trained Word2Vec or FastText src to find synonyms
word_vectors = KeyedVectors.load_word2vec_format("word2vec_healthcare.bin", binary=True)


def map_labels_to_ids(label):
    """
        Maps labels to IDs in the graph.
        The labels are taken from MIMIC_LABELS variable in settings.py.
        The IDs are mapped from 0 to 13 directly.
        If the graph uses a different mapping use the function map_labels_to_ids_from_graph(graph_json).
        :return: the mapped ID of the label as string.
        :rtype: str
    """

    label_to_id = {
        "Atelectasis": 0,
        "Cardiomegaly": 1,
        "Consolidation": 2,
        "Edema": 3,
        "Enlarged Cardiomediastinum": 4,
        "Fracture": 5,
        "Lung Lesion": 6,
        "Lung Opacity": 7,
        "No Finding": 8,
        "Pleural Effusion": 9,
        "Pleural Other": 10,
        "Pneumonia": 11,
        "Pneumothorax": 12,
        "Support Devices": 13
    }
    return str(label_to_id.get(label, -1))  # Return -1 if label not found


def map_labels_to_ids_from_graph(graph_json, label):
    """
        Maps labels to IDs in the graph.
        The labels are taken from MIMIC_LABELS variable in settings.py.
        The IDs are mapped from 0 to 13 directly.
        If the graph uses a different mapping use the function map_labels_to_ids(graph_json).
        :param graph_json: JSON graph data.
        :type graph_json: dict
        :param label: Label to map.
        :type label: str
        :return: the mapped ID of the label as string.
        :rtype: str
    """
    label_to_id = {}
    for node in graph_json["nodes"]:
        label_to_id[node["label"]] = node["id"]

    return str(label_to_id.get(label, -1))  # Return -1 if label not found


def extract_keywords(text, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Extract keywords from a medical report using SciSpacy + Word2Vec.

    Args:
        text (str): Medical report to analyze.
        similarity_threshold (float): Threshold for synonym similarity.

    Returns:
        list: List of keywords extracted from the report.
    """
    doc = nlp(text)
    keywords = set()

    # Named Entity Recognition (NER) with SciSpacy
    for ent in doc.ents:
        keywords.add(ent.text.lower())

    # If a word is not found, look for synonyms with Word2Vec
    for token in doc:
        if token.text.lower() not in keywords and token.text in word_vectors:
            similar_words = word_vectors.most_similar(token.text, topn=3)
            for word, similarity in similar_words:
                if similarity > similarity_threshold:  # Similarity threshold
                    keywords.add(word.lower())

    return list(keywords)


def match_keywords_to_graph(keywords, graph_json):
    """
    Maps keywords extracted from the report to nodes in the graph.
    Args:
        keywords (list): List of keywords extracted from the report.
        graph_json (dict): JSON graph data.
    Returns:
        dict: Mapping of keywords to graph nodes.
    """
    matched_nodes = {}

    for keyword in keywords:
        # Here we use difflib to find the closest match
        # Because in training use of nlp would slow down the process excessively
        for node in graph_json["nodes"]:
            possible_matches = [node["label"]] + node.get("synonyms", [])
            closest = get_close_matches(keyword, possible_matches, n=1, cutoff=0.6)

            if closest:
                matched_nodes[keyword] = node["id"]

    return matched_nodes


# ========================================================== EDGES VALUES ATTRIBUTION =========================================================

def update_edges_with_keywords(graph_json, labels, keywords):
    """
    For each 'finding' edge connecting a disease node and a sign node:
      - If the disease node is positive (value >= 1.0) and the sign node is matched in the report keywords,
        add POSITIVE_WEIGHT_FINDING.
      - If the disease node is positive but the sign node is not matched,
        subtract NEGATIVE_WEIGHT_FINDING.
    This update is made regardless of the source/target ordering.
    """
    # Build a mapping from node id to node info for easy lookup.
    node_map = {node["id"]: node for node in graph_json["nodes"]}

    # Compute keyword-to-node matching once.
    matchings = match_keywords_to_graph(keywords, graph_json)
    matched_ids = set(matchings.values())

    # Process only finding edges.
    finding_edges = [edge for edge in graph_json["edges"] if edge.get("type") == "finding"]

    for edge in finding_edges:
        node1 = node_map.get(edge["source"])
        node2 = node_map.get(edge["target"])
        if not node1 or not node2:
            continue  # Skip if nodes are missing

        # Determine which node is the disease (label) and which is the clinical finding (sign)
        if node1.get("type") == "disease" and node2.get("type") == "sign":
            disease_node, sign_node = node1, node2
        elif node1.get("type") == "sign" and node2.get("type") == "disease":
            disease_node, sign_node = node2, node1
        else:
            # If the types are not clearly differentiated, skip updating this edge.
            continue

        # Check if the disease node is positive in ground truth.
        if labels.get(disease_node["id"], 0.0) >= 1.0:
            if sign_node["id"] in matched_ids:
                edge["weight"] += POSITIVE_WEIGHT_FINDING
            else:
                edge["weight"] -= NEGATIVE_WEIGHT_FINDING

    return graph_json


def __update_edges_with_keywords__(graph_json, labels, keywords):
    """
    Update the edges of the graph with the keywords extracted from the report.
    Args:
        graph_json (dict): JSON graph data.
        labels (dict): Dictionary with:
         -key: (str) node IDs
         -value: (float) is the value of the node for a specific xr, 1.0 if present, 0.0 if not present.
        keywords (list): List of keywords IDs (as string) extracted from the report.
    Returns:
        dict: Updated graph with edges containing keywords.
    """

    # For each positive label in ground truth, check if findings are present in the report
    for label in labels:
        if labels[label] < 1.0:
            continue
        # Check if the label is in the keywords
        matchings = match_keywords_to_graph(keywords, graph_json)

        finding_edges = [edge for edge in graph_json["edges"] if edge.get("type") == "finding"]

        for edge in finding_edges:
            if edge["source"] == label and edge["target"] in matchings.values():
                edge["weight"] += POSITIVE_WEIGHT_FINDING # Increase weight if both nodes are in the report
            elif edge["source"] == label and edge["target"] not in matchings.values():
                edge["weight"] -= NEGATIVE_WEIGHT_FINDING

    return graph_json


def update_correlation_edges(graph_json, labels):
    """
    Update the 'correlation' edges:
      - If both nodes (disease labels) are positive, add POSITIVE_WEIGHT_CORR.
      - If either is not positive, subtract NEGATIVE_WEIGHT_CORR.
    Args:
        graph_json (dict): JSON graph data.
        labels (dict): Dictionary with:
         -key: (str) node IDs
         -value: (float) is the value of the node for a specific xr, 1.0 if present, 0.0 if not present.
    Returns:
        dict: Updated graph with edges containing correlation.
    """
    # Get edges of type "correlation"
    correlation_edges = [edge for edge in graph_json["edges"] if edge.get("type") == "correlation"]

    # Positive +1.0, Negative 0.0 (in label ground truth)
    for correlation in correlation_edges:
        # Get the source and target node IDs
        # Here we assume that the source and target IDs are mapped in labels by position
        source_value = labels.get(correlation["source"], 0.0)
        target_value = labels.get(correlation["target"], 0.0)

        if source_value > 0 and target_value > 0:
            correlation["weight"] += POSITIVE_WEIGHT_CORR
        elif source_value == 0 or target_value == 0:
            correlation["weight"] -= NEGATIVE_WEIGHT_CORR

    return graph_json