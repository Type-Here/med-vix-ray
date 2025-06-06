"""
    This module contains functions to update the edges of a graph with keywords extracted from reports.
    Each function is designed to handle specific tasks related to graph manipulation and keyword matching
    or update edges weights with different strategies.
"""


# Note gt_entry is a dictionary with the ground truth entries for each finding with structure:
#{ "sign_node_id": [s_gt, polarity], # s_gt is the similarity score, polarity is 1 for positively and 0 for negatively correlated to the image
#  "sign_node_id_2": [s_gt, polarity],
#       ...,
#  "dicom_ids": [dicom_id1, dicom_id2, ...],
#}, ...


def bayes_b_distribution(graph, positive_disease_ids, gt_entry):
    """
    Update edge weights in the graph using Beta distribution per edge.
    We model each edge weight as the expected value of a Beta(α, β) distribution:
        E[weight] = α / (α + β)

    - α = number of positive observations
    - β = number of negative (or missing) observations

    Args:
        graph (dict): The graph structure with "nodes" and "links"
        positive_disease_ids (list): List of positive disease IDs
        gt_entry (dict): Dictionary with ground truth entries for each finding

    Returns:
        Updated graph_json with per-edge {alpha, beta, weight}
    """

    for edge in graph["links"]:
        #if edge["relation"] not in ("correlation", "finding"):
        #    continue

        # Init Beta parameters if missing
        edge.setdefault("alpha", 1)  # 1 prior positive
        edge.setdefault("beta", 1)  # 1 prior negative

        if edge["relation"] == "correlation":
            d1, d2 = int(edge["source"]), int(edge["target"])
            is_positive = d1 in positive_disease_ids and d2 in positive_disease_ids
            s_gt = 1.0

        #elif edge["relation"] == "finding":
        else:
            s_idx = int(edge["source"])
            t_idx = int(edge["target"])
            if s_idx not in positive_disease_ids:
                continue  # skip this edge, disease is not present

            # Check if sign is mentioned in NER
            entry = gt_entry.get(str(t_idx), [0.0, -1])
            if entry[1] == -1:
                is_positive = False  # missing = treat as negative
                s_gt = 0.5
            else:
                s_gt, polarity = entry
                s_gt = max(0.0, min(1.0, s_gt))  # clamp in [0, 1]
                is_positive = polarity == 1

        # Update parameters
        if is_positive:
            edge["alpha"] += min(s_gt + 0.2, 1.0)  # add a small positive value
        else:
            edge["beta"] += min(s_gt + 0.1, 1.0)  # add a small positive value

        # Update the estimated weight
        alpha, beta = edge["alpha"], edge["beta"]
        edge["weight"] = alpha / (alpha + beta)

    return graph


def naive_median_count(graph_json, positive_labels, gt_entry):
    """
    Update the edge weights in the graph using a naive approach: median count of positive labels.
    Args:
        graph_json (dict): The graph structure with "nodes" and "links"
        positive_labels (set): Set of positive disease IDs
        gt_entry (dict): Dictionary with ground truth entries for each finding
    Returns:
        dict: Updated graph_json with per-edge {weight, update_count}
    """
    for edge in graph_json["links"]:
        source = int(edge["source"])  # disease node index
        target = int(edge["target"])  # sign node index

        if edge["relation"] == "correlation":
            # If both disease nodes are present in the positive labels,
            # update the weight positively.
            if source not in positive_labels or target not in positive_labels:
                continue

            # If both disease nodes are positive, add a positive weight.
            polarity = 1.0

            weight_old = edge.get("weight", 1.0)
            count = edge.get("update_count", 0)

            # Compute the new weight with weighted moving average.
            weight_new = (count * weight_old + polarity) / (count + 1)
            edge["weight"] = weight_new
            edge["update_count"] = count + 1


        elif edge["relation"] == "finding":
            disease_idx = int(edge["source"])  # disease node index

            # If the disease node is not in the positive labels, skip this edge.
            if disease_idx not in positive_labels:
                continue

            sign_idx = int(edge["target"])  # sign node index
            weight_old = edge.get("weight", 1.0)
            count = edge.get("update_count", 0)

            # Determine the new similarity value s_new.
            # Check if the ground truth indicates anything for this sign node.
            if str(sign_idx) in gt_entry:
                s_gt, polarity = gt_entry[str(sign_idx)]
                s_new = s_gt if polarity == 1 else 0.2
            else:
                # If not mentioned, assume the sign was not observed.
                # TODO: Check this value
                s_new = 0.4

            # Compute the new weight with weighted moving average.
            weight_new = (count * weight_old + s_new) / (count + 1)
            edge["weight"] = weight_new
            edge["update_count"] = count + 1

    return graph_json
