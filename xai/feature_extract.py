import numpy as np
import cv2
from scipy.stats import entropy, skew, kurtosis


def extract_heatmap_features(map, threshold=0.5):
    """
    Extract numerical features from the attention heatmap of the model.

    Args:
        map (np.array): Attention Map or Heatmap (256x256).
        threshold (float): Threshold over which get data from.

    Returns:
        dict: Dictionary of the extracted features.
    """
    # Normalize the heatmap between 0 and 1
    heatmap = map / np.max(map)

    # Mean Intensity and Variance
    mean_intensity = np.mean(heatmap)
    variance_intensity = np.var(heatmap)

    # Calculate entropy of the image
    hist, _ = np.histogram(heatmap, bins=256, range=(0, 1))
    entropy_value = entropy(hist)

    # In order to calculate the activated area, we binarize the heatmap
    binary_map = (heatmap > threshold).astype(np.uint8)
    activated_area = np.sum(binary_map) / heatmap.size  # Percentage of activated area

    # Flatten the heatmap for skewness and kurtosis calculation
    skewness_value = skew(heatmap.flatten())
    kurtosis_value = kurtosis(heatmap.flatten())

    # Fractal dimension calculation (box-counting method)
    def fractal_dimension(z, f_threshold=0.9):
        """
            Calculate the fractal dimension using the box-counting method.
            Args:
                z (np.array): Binary map.
                f_threshold (float): Threshold for fractal dimension calculation.
        """
        z = z < f_threshold
        sizes = 2 ** np.arange(1, 8)
        counts = []
        for size in sizes:
            counts.append(np.sum(cv2.resize(z.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)))
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
        "fractal": fractal_value
    }

def update_weighted_mean(prev_mean, new_value, count):
    """
    Update the weighted mean with a new value.
    Args:
        prev_mean (float): Previous mean value.
        new_value (float): New value to add.
        count (int): Total number of observations so far.

    Returns:
        float: Updated mean value.
    """
    return (count * prev_mean + new_value) / (count + 1)



def update_graph_features(graph_json, extracted_features, sign_label):
    """
    Updates the features of the nodes in the JSON graph during training.
    Args:
        graph_json (dict): JSON graph data.
        extracted_features (dict): Heatmap features extracted from the attention map.
        sign_label (str): Name of the radiological sign to update.
    """
    for node in graph_json["nodes"]:
        if node["label"] == sign_label:
            for key, value in extracted_features.items():
                if node["features"][key] is None:
                    # If First observation
                    node["features"][key] = value
                else:
                    # Apply the weighted moving average
                    node["features"][key] = update_weighted_mean(
                        node["features"][key], value, node["count"]
                    )

                node["count"] += 1  # TODO: Update the count of observations: choose if count in json or in memory
    return graph_json

# ============================

