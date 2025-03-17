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


def update_weighted_mean(prev_mean, new_value, count):
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


def update_graph_features(graph_json, extracted_features, sign_label):
    """
    Update the clinical finding node in the JSON graph with newly extracted features.

    For each node with label == sign_label, update each feature using a weighted moving average.
    Special handling is done for the "position" feature (a list of 4 values).

    Args:
        graph_json (dict): JSON graph data.
        extracted_features (dict): Features extracted from the attention heatmap.
        sign_label (str): Radiological sign (clinical finding) label to update.

    Returns:
        dict: Updated graph JSON.
    """
    for node in graph_json["nodes"]:
        if node["label"] == sign_label:
            # Ensure the node has a count of observations (initialize if missing)
            if "count" not in node:
                node["count"] = 0

            for key, value in extracted_features.items():
                if key == "position":
                    # Update the bounding box as a vector (each coordinate updated separately)
                    if node["features"].get(key) is None or node["count"] == 0:
                        node["features"][key] = value
                    else:
                        updated_position = []
                        for i in range(4):
                            updated_position.append(
                                update_weighted_mean(node["features"][key][i], value[i], node["count"])
                            )
                        node["features"][key] = updated_position
                else:
                    # For scalar features
                    if node["features"].get(key) is None:
                        node["features"][key] = value
                    else:
                        node["features"][key] = update_weighted_mean(
                            node["features"][key], value, node["count"]
                        )
            node["count"] += 1  # Increase the observation count
    return graph_json
