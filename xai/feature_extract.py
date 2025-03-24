import numpy as np
import cv2
from scipy.stats import entropy, skew, kurtosis


def extract_heatmap_features(att_map, threshold=0.5):
    """
    Extract numerical features from an attention heatmap (e.g., from CDAM or a modified Grad-CAM).

    Args:
        att_map (np.array): Attention map or heatmap (e.g., 256x256).
        threshold (float): Threshold for binarizing the heatmap.

    Returns:
        dict: Dictionary of extracted features, including:
              - intensity: Mean intensity.
              - variance: Variance of intensities.
              - entropy: Shannon entropy of the normalized histogram.
              - active_dim: Fraction of the heatmap activated above the threshold.
              - skewness: Skewness of the intensity distribution.
              - kurtosis: Kurtosis of the intensity distribution.
              - fractal: Fractal dimension computed with a box-counting method.
              - position: [x_min, x_max, y_min, y_max] bounding box of the activated area.
    """
    # Normalize the heatmap between 0 and 1
    heatmap = att_map / np.max(att_map)

    # Mean intensity and variance
    mean_intensity = np.mean(heatmap)
    variance_intensity = np.var(heatmap)

    # Calculate entropy using histogram of pixel intensities
    hist, _ = np.histogram(heatmap, bins=256, range=(0, 1))
    entropy_value = entropy(hist)

    # Binarize the heatmap to identify activated regions
    binary_map = (heatmap > threshold).astype(np.uint8)
    activated_area = np.sum(binary_map) / heatmap.size  # Fraction of activated area

    # Determine the bounding box (position) of the activated region
    activated_indices = np.where(binary_map > 0)
    if activated_indices[0].size > 0:
        # Note: np.where returns (rows, cols) => (y, x)
        y_min, y_max = int(np.min(activated_indices[0])), int(np.max(activated_indices[0]))
        x_min, x_max = int(np.min(activated_indices[1])), int(np.max(activated_indices[1]))
    else:
        # If no activation is found, default to zeros
        x_min, x_max, y_min, y_max = 0, 0, 0, 0

    # Calculate skewness and kurtosis on the flattened heatmap
    skewness_value = skew(heatmap.flatten())
    kurtosis_value = kurtosis(heatmap.flatten())

    # Fractal dimension calculation using the box-counting method.
    # Here, we use a binary map with a threshold trick.
    def fractal_dimension(z, f_threshold=0.9):
        """
        Calculate fractal dimension using box-counting.

        Args:
            z (np.array): Binary map.
            f_threshold (float): Threshold value for computing the box counts.
        """
        # Convert binary map to boolean mask
        z_bool = z < f_threshold
        sizes = 2 ** np.arange(1, 8)
        counts = []
        for size in sizes:
            # Resize using nearest-neighbor interpolation to count boxes
            resized = cv2.resize(z_bool.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
            counts.append(np.sum(resized))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    fractal_value = fractal_dimension(binary_map)

    return {
        "intensity": mean_intensity,
        "variance": variance_intensity,
        "entropy": entropy_value,
        "active_dim": activated_area,
        "skewness": skewness_value,
        "kurtosis": kurtosis_value,
        "fractal": fractal_value,
        "position": [x_min, x_max, y_min, y_max]
    }


def _compute_feature_vector(features, keys=None):
    """
    Convert a dictionary of features into a vector.
    If keys is None, use a default list.
    Args:
        features (dict): Dictionary of features.
        keys (list): List of keys to extract from the dictionary.
    Returns:
        np.array: Features vectorized as a numpy array.
    """
    if keys is None:
        keys = ["intensity", "variance", "skewness", "kurtosis", "active_dim", "entropy", "fractal"]
    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)


def __cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two numpy vectors.
    Returns a scalar between -1 and 1.
    Args:
        vec1 (np.array): First vector.
        vec2 (np.array): Second vector.
    Returns:
        float: Cosine similarity between the two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_sign_bias(graph_json, num_diseases):
    """
    Compute an extra bias vector for disease nodes based on the connected sign nodes.
    Returns a numpy array of shape (num_diseases,).
    Note:
        - num_diseases is equal to the number of labels of the model.

    Note:
        - Assumes that the graph_json contains edges of type 'finding' and nodes of type 'sign'.
        - It also assumes that findings relations have source as disease and target as sign.

    Args:
        graph_json (dict): JSON graph data.
        num_diseases (int): Number of disease nodes.

    Returns:
        np.array: Bias vector for disease nodes.
    """
    # Initialize bias vector for diseases.
    bias = np.zeros(num_diseases, dtype=np.float32)

    # Iterate over edges of type 'finding'.
    for edge in graph_json["edges"]:
        if edge["type"] == "finding":
            disease_idx = int(edge["source"])  # disease index
            sign_idx = int(edge["target"])  # sign index
            weight = edge.get("weight", 1.0)
            # Find the corresponding sign node in graph_json["nodes"].
            # We assume that sign nodes have an attribute "type"=="sign"
            for node in graph_json["nodes"]:
                if node.get("type") == "sign" and int(node["id"]) == sign_idx:
                    similarity = node.get("similarity", 1.0)
                    # Accumulate the contribution.
                    bias[disease_idx] += weight * similarity
                    break  # found the sign node, move to next edge
    return bias  # shape (num_diseases,)


def __update_weighted_mean(prev_mean, new_value, count):
    """
    Update a running mean using a weighted moving average.

    Args:
        prev_mean (float): Previous mean value.
        new_value (float): New observation.
        count (int): Number of observations before the new one.

    Returns:
        float: Updated mean value.
    """
    return (count * prev_mean + new_value) / (count + 1)


def _similarity_evaluation_single_node(node, extracted_features):
    """
        Compute the similarity between the stored features of a node and the extracted features.

        Args:
            node (dict): Node from the JSON graph with features.
            extracted_features (dict): Features extracted from the attention heatmap.
    """
    stored_vec = _compute_feature_vector(node["features"])
    extracted_vec = _compute_feature_vector(extracted_features)
    sim = __cosine_similarity(stored_vec, extracted_vec)
    sim = (sim + 1.0) / 2.0  # normalize to [0,1]
    # You can decide whether to update node["similarity"] or leave it as is.
    node["similarity"] = sim


def _similarity_evaluation(graph_json, extracted_features):
    """
    Compute the similarity between the stored features of each node and the extracted features.

    Args:
        graph_json (dict): JSON graph data.
        extracted_features (dict): Features extracted from the attention heatmap.

    Returns:
        dict: Updated graph JSON with similarity scores for each node.
    """
    for node in graph_json["nodes"]:
        if node.get("type") == "sign":
            _similarity_evaluation_single_node(node, extracted_features)
    return graph_json


def update_graph_features(graph_json, extracted_features, sign_label, apply_similarity=False):
    """
    Update the clinical finding node in the JSON graph with newly extracted features.

    For each node with label == sign_label, update each feature using
    a weighted moving average.
    Special handling is done for the "position" feature (a list of 4 values).

    Args:
        graph_json (dict): JSON graph data.
        extracted_features (dict): Features extracted from the attention heatmap.
        sign_label (str): Radiological sign (clinical finding) label to update.
        apply_similarity (bool): If True, compute similarity between the node
            and the extracted features.

    Returns:
        dict: Updated graph JSON.
    """
    for node in graph_json["nodes"]:
        if node["label"] == sign_label:
            # Ensure the node has a count of observations (initialize if missing)
            if "count" not in node:
                node["count"] = 0

            for key, value in extracted_features.items():
                if node["features"].get(key) is None or node["count"] == 0:
                    # Initialize the feature if it doesn't exist or if this is the first observation
                    node["features"][key] = value

                elif key == "position":
                    # Update the bounding box as a vector (each coordinate updated separately)
                    node["features"][key] = [__update_weighted_mean(node["features"][key][i],
                                                                    value[i], node["count"]) for i in range(4)]
                else:
                    # For scalar features
                    node["features"][key] = __update_weighted_mean(
                        node["features"][key], value, node["count"]
                    )
            # Update the observation count
            node["count"] += 1

            # If apply_similarity is True, compute the similarity for this node
            if apply_similarity:
                _similarity_evaluation_single_node(node, extracted_features)
    return graph_json