import numpy as np
import torch
import cv2
from scipy.stats import skew, kurtosis
from settings import SIMILARITY_THRESHOLD


def extract_heatmap_features_multiregion(att_maps, threshold=SIMILARITY_THRESHOLD):
    """
    Process a batch of attention maps and extract features for each image using extract_sign_features
    with multi-region recognition.
    This should handle both single and batch inputs.

    Args:
        att_maps (np.array or torch.Tensor): Batch of attention maps with shape [B, H, W].
        threshold (float): Threshold for binarization.

    Returns:
        dict: A dictionary where each key maps to a tensor of shape [B] (or [B, 4] for 'position').
        It's a combination of dictionaries for each image in batch.

    Note:
        dict features: Dictionary of extracted features, including:
              - intensity: Mean intensity.
              - variance: Variance of intensities.
              - entropy: Shannon entropy of the normalized histogram.
              - active_dim: Fraction of the heatmap activated above the threshold.
              - skewness: Skewness of the intensity distribution.
              - kurtosis: Kurtosis of the intensity distribution.
              - fractal: Fractal dimension computed with a box-counting method.
              - position: [x_min, x_max, y_min, y_max] bounding box of the activated area.
    """
    # If att_maps is a torch.Tensor, convert to numpy array for processing.
    if isinstance(att_maps, torch.Tensor):
        att_maps = att_maps.cpu().numpy()
    B = att_maps.shape[0]
    feature_dicts = []
    for i in range(B):
        feature_dicts.append(__extract_heatmap_features_single_map_multiregion(att_maps[i], threshold))

    # Combine the per-sample dictionaries into batched tensors.
    combined_features = {}
    for key in feature_dicts[0]:
        # For keys where the value is a scalar, stack into a tensor of shape [B].
        # For 'position', values are lists of 4 numbers, so stack into shape [B, 4].
        combined_features[key] = torch.tensor([d[key] for d in feature_dicts], dtype=torch.float32)
    return combined_features


def extract_heatmap_features(att_map, threshold=SIMILARITY_THRESHOLD):
    """
        Extract numerical features from an attention heatmap (e.g., from CDAM or a modified Grad-CAM).
        This is the main function to call for feature extraction.
        It handles both single and batch inputs.

        Args:
            att_map (np.array): Attention map or heatmap (e.g., 256x256).
            threshold (float): Threshold for binarizing the heatmap.

        Returns:
            dict: If input is a single map, returns a dictionary of extracted features.
            If input is a batch, returns a dictionary of extracted features in multiple shapes.

        Note:
            dict features: Dictionary of extracted features, including:
                  - intensity: Mean intensity.
                  - variance: Variance of intensities.
                  - entropy: Shannon entropy of the normalized histogram.
                  - active_dim: Fraction of the heatmap activated above the threshold.
                  - skewness: Skewness of the intensity distribution.
                  - kurtosis: Kurtosis of the intensity distribution.
                  - fractal: Fractal dimension computed with a box-counting method.
                  - position: [x_min, x_max, y_min, y_max] bounding box of the activated area.
        """
    # If input is a batch:
    if att_map.ndim == 3:
        b_size = att_map.shape[0]
        feature_dicts = []
        for i in range(b_size):
            feature_dicts.append(extract_heatmap_features(att_map[i], threshold))
        # Combine the dictionaries: for each key, stack values into a tensor.
        combined_features = {}
        for key in feature_dicts[0]:
            combined_features[key] = torch.tensor([d[key] for d in feature_dicts], dtype=torch.float32)
        return combined_features
    else:
        # If input is a single map, extract features.
        return __extract_heatmap_features_single_map(att_map, threshold)


def __fractal_dimension(z, f_threshold=0.9):
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
        resized = cv2.resize(z_bool.astype(np.uint8), (size, size),
                             interpolation=cv2.INTER_NEAREST)
        counts.append(np.sum(resized))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def __extract_heatmap_features_single_map_multiregion(att_map, threshold=SIMILARITY_THRESHOLD):
        """
        Extract features from an attention map that may contain multiple distinct sign regions.

        Instead of computing global statistics, this function:
          - Normalizes the attention map.
          - Thresholds the map to produce a binary image.
          - Uses connected component analysis to identify candidate regions.
          - For each candidate region, computes statistical features (mean, variance, entropy,
            active area, skewness, kurtosis, fractal dimension, and bounding box).
          - Selects a representative region (here, the region with the largest area).

        If no region is found, it falls back to global extraction.

        Args:
            att_map (np.array): A 2D attention map (shape: [H, W]). Can be unnormalized.
            threshold (float): Threshold for binarization.

        Returns:
            dict: A dictionary containing the extracted features from the selected region.
                  Keys include: "intensity", "variance", "entropy", "active_dim",
                  "skewness", "kurtosis", "fractal", "position" (bounding box as [x_min, x_max, y_min, y_max]).
        """
        # Ensure the attention map is float32.
        att_map = att_map.astype(np.float32)
        # Normalize the map to [0,1] if needed.
        max_val = np.max(att_map)
        heatmap = att_map / max_val if max_val > 1.0 else att_map

        # Binarize the heatmap.
        binary_map = (heatmap > threshold).astype(np.uint8)

        # Use connected component analysis to detect distinct regions.
        # This will label each connected component with a unique integer.
        # The background will be labeled as 0.
        #
        # Here is the main difference with the single map extraction:
        num_labels, labels_im, stats, centroids = (
            cv2.connectedComponentsWithStats(binary_map, connectivity=8))

        # Initialize a list to hold region features.
        region_features = []
        for label in range(1, num_labels):  # label 0 is background.
            # Create a mask for the region.
            region_mask = (labels_im == label).astype(np.uint8)
            region_area = stats[label, cv2.CC_STAT_AREA]

            # Compute region-specific statistics.
            region_pixels = heatmap[labels_im == label]
            region_intensity = np.mean(region_pixels)
            region_variance = np.var(region_pixels)

            # Entropy
            hist, _ = np.histogram(region_pixels, bins=256, range=(0, 1))
            hist = hist + 1e-6  # avoid log(0)
            prob = hist / np.sum(hist)
            region_entropy = -np.sum(prob * np.log(prob))

            # Activated area fraction relative to full image.
            region_active_dim = region_area / heatmap.size

            # Skewness and kurtosis.
            region_skewness = skew(region_pixels.flatten())
            region_kurtosis = kurtosis(region_pixels.flatten())

            # Fractal dimension using box-counting.
            region_fractal = __fractal_dimension(region_mask)

            # Bounding box using stats: [x, y, width, height]
            x, y, w, h = (stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP],
                          stats[label, cv2.CC_STAT_WIDTH],
                          stats[label, cv2.CC_STAT_HEIGHT]
                          )
            # Normalize the bounding box coordinates from 0 to 1
            x_min, x_max = x / heatmap.shape[1], (x + w) / heatmap.shape[1]
            y_min, y_max = y / heatmap.shape[0], (y + h) / heatmap.shape[0]

            region_position = [x_min, x_max, y_min, y_max]

            region_features.append({
                "intensity": region_intensity,
                "variance": region_variance,
                "entropy": region_entropy,
                "active_dim": region_active_dim,
                "skewness": region_skewness,
                "kurtosis": region_kurtosis,
                "fractal": region_fractal,
                "position": region_position,
                "area": region_area
            })

        # If no region is found, fall back to global extraction.
        if len(region_features) == 0:
            return __extract_heatmap_features_single_map(att_map, threshold)

        # Select the region with the largest area as representative.
        selected_region = max(region_features, key=lambda r: r["area"])
        # Optionally remove the 'area' key if not needed.
        selected_region.pop("area", None)
        return selected_region


def __extract_heatmap_features_single_map(att_map, threshold=SIMILARITY_THRESHOLD):
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
    # Ensure the attention map is a float32 array.
    att_map = att_map.cpu().numpy().astype(np.float32)
    # or keep pyTorch
    # att_map = att_map.float()

    # If the map's max value is above 1, normalize to [0,1].
    max_val = np.max(att_map)
    if max_val > 1.0:
        heatmap = att_map / max_val
    else:
        heatmap = att_map

    # Compute mean intensity and variance
    mean_intensity = np.mean(heatmap)
    variance_intensity = np.var(heatmap)

    # Calculate entropy using histogram of pixel intensities
    hist, _ = np.histogram(heatmap, bins=256, range=(0, 1))
    hist = hist + 1e-6  # add epsilon to avoid log(0)
    prob = hist / np.sum(hist)
    entropy_value = -np.sum(prob * np.log(prob))
    # entropy_value = entropy(hist)

    # Binarize the heatmap to identify activated regions
    binary_map = (heatmap > threshold).astype(np.uint8)
    activated_area = np.sum(binary_map) / heatmap.size  # Fraction of activated area

    # Determine the bounding box (position) of the activated region
    activated_indices = np.where(binary_map > 0)
    if activated_indices[0].size > 0:
        # Note: np.where returns (rows, cols) => (y, x)
        y_min, y_max = int(np.min(activated_indices[0])), int(np.max(activated_indices[0]))
        x_min, x_max = int(np.min(activated_indices[1])), int(np.max(activated_indices[1]))
        # Normalize the values from 0 to 1
        y_min, y_max = y_min / heatmap.shape[0], y_max / heatmap.shape[0]
        x_min, x_max = x_min / heatmap.shape[1], x_max / heatmap.shape[1]
    else:
        # If no activation is found, default to zeros
        x_min, x_max, y_min, y_max = 0, 0, 0, 0

    # Calculate skewness and kurtosis on the flattened heatmap
    skewness_value = skew(heatmap.flatten())
    kurtosis_value = kurtosis(heatmap.flatten())

    # Fractal dimension calculation using the box-counting method.
    # Here, we use a binary map with a threshold trick.
    fractal_value = __fractal_dimension(binary_map)

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
    for edge in graph_json["links"]:
        if edge["relation"] == "finding":
            disease_idx = int(edge["source"])  # disease index
            sign_idx = int(edge["target"])  # sign index
            weight = edge.get("weight", 1.0)
            # Find the corresponding sign node in graph_json["nodes"].
            # We assume that sign nodes have an attribute "type"=="sign"
            for node in graph_json["nodes"]:
                if node.get("relation") == "sign" and int(node["id"]) == sign_idx:
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


def update_graph_features(graph, extracted_features, sign_label, apply_similarity=False):
    """
    Update the clinical finding node in the JSON graph with newly extracted features.
    It supports both single-sample and batched inputs.

    For each node with label == sign_label, update each feature using a weighted moving average.
    Special handling is done for the "position" feature (a list of 4 values).

    Args:
        graph (dict): JSON graph data.
        extracted_features (dict or dict of torch.Tensor):
            - If single-sample, a dictionary of features (e.g., {"intensity": float, "position": [x_min,x_max,y_min,y_max], ...}).
            - If batched, a dictionary where each key maps to a torch.Tensor of shape [B] (or [B, 4] for position).
        sign_label (str): Radiological sign (clinical finding) label to update.
        apply_similarity (bool): If True, compute similarity between the node and the extracted features.

    Returns:
        dict: Updated graph_json.
    """
    # Detect batch or single-sample
    is_batch = any(isinstance(v, torch.Tensor) and v.ndim >= 1 for v in extracted_features.values())

    if is_batch:
        batch_size = next(iter(extracted_features.values())).shape[0]
        for i in range(batch_size):
            single = {
                k: v[i].item() if v.ndim == 1 else v[i].tolist()
                for k, v in extracted_features.items()
            }
            graph = update_graph_features(graph, single, sign_label, apply_similarity)
        return graph

    # Now processing a single sample
    for node in graph["nodes"]:
        if node.get("label") != sign_label:
            continue

        # Initialize observation count and feature dict
        node.setdefault("count", 0)
        #node.setdefault("features", {})

        for key, value in extracted_features.items():
            if value is None:
                continue

            # Init feature if not present
            if key not in node["features"] or node["count"] == 0:
                node["features"][key] = value
            elif key == "position":
                # Update each of the 4 coordinates
                node["features"][key] = [
                    __update_weighted_mean(node["features"][key][i], value[i], node["count"])
                    for i in range(4)
                ]
            else:
                node["features"][key] = __update_weighted_mean(
                    node["features"][key], value, node["count"]
                )

        # Optional similarity evaluation
        if apply_similarity:
            _similarity_evaluation_single_node(node, extracted_features)

        node["count"] += 1  # Only if actually updated

    return graph