import numpy as np
import torch
from scipy.stats import skew, kurtosis
from settings import SIMILARITY_THRESHOLD

def extract_attention_batch(attn_maps: torch.Tensor, device: torch.device):
    """
    Process a batch of attention maps and extract compact statistical features.

    Args:
        attn_maps (torch.Tensor): Batch of attention maps with shape [B, num_heads, num_tokens, num_tokens].
        device (torch.device): Device on which to perform computation.

    Returns:
        torch.Tensor: Feature tensor of shape [B, 4], where each feature vector corresponds to:
                      [entropy, skewness, center_x, center_y].
    """
    batch_size, num_heads, num_tokens, _ = attn_maps.shape

    # 1. Average over heads
    attn_maps = attn_maps.mean(dim=1)  # [B, num_tokens, num_tokens]

    # 2. Extract attention from CLS token to patches
    attn_maps = attn_maps[:, 0, 1:]  # [B, num_patches]

    # 3. Reshape to spatial grid
    side = int(num_tokens ** 0.5)
    attn_maps = attn_maps.reshape(batch_size, side, side)  # [B, side, side]

    # 4. Normalize attention maps to [0, 1]
    attn_maps_min = attn_maps.flatten(1).min(dim=1, keepdim=True).unsqueeze(-1)
    attn_maps_max = attn_maps.flatten(1).max(dim=1, keepdim=True).unsqueeze(-1)
    attn_maps = (attn_maps - attn_maps_min) / (attn_maps_max - attn_maps_min + 1e-6)

    # 5. Compute entropy
    entropy = -(attn_maps * torch.log(attn_maps + 1e-6)).sum(dim=[1, 2])

    # 6. Compute skewness
    flat_maps = attn_maps.flatten(1)
    mean = flat_maps.mean(dim=1, keepdim=True)
    std = flat_maps.std(dim=1, keepdim=True)
    skewness = ((flat_maps - mean) ** 3).mean(dim=1) / (std.squeeze() ** 3 + 1e-6)

    # 7. Compute center of mass
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(0, 1, side, device=device),
        torch.linspace(0, 1, side, device=device),
        indexing='ij'
    )
    mass = attn_maps.sum(dim=[1, 2]) + 1e-6
    center_x = (attn_maps * grid_x.unsqueeze(0)).sum(dim=[1, 2]) / mass
    center_y = (attn_maps * grid_y.unsqueeze(0)).sum(dim=[1, 2]) / mass

    # 8. Stack all features
    features = torch.stack([entropy, skewness, center_x, center_y], dim=1)  # [B, 4]

    return features


def extract_attention_batch_multiregion(attn_maps: torch.Tensor, device: torch.device, threshold: float = 0.6):
    """
    Extract all clinically-relevant attention-based features for each image in batch.
    Matches the structure expected by the graph ("features" field in sign nodes).

    Args:
        attn_maps (torch.Tensor): Attention maps [B, num_heads, T, T].
        device (torch.device): Device.
        threshold (float): Binarization threshold.

    Returns:
        dict: {
            "intensity": [B],
            "variance": [B],
            "entropy": [B],
            "active_dim": [B],
            "skewness": [B],
            "kurtosis": [B],
            "fractal": [B],
            "position": [B, 4] (x_min, x_max, y_min, y_max)
        }
    """
    B, H, T, _ = attn_maps.shape
    attn_maps = attn_maps.mean(dim=1)  # [B, T, T]
    attn_maps = attn_maps[:, 0, 1:]  # [B, T-1]

    side = int((T - 1) ** 0.5)
    attn_maps = attn_maps.reshape(B, side, side)  # [B, H, W]

    attn_maps_np = attn_maps.detach().cpu().numpy()
    features = {
        "intensity": [],
        "variance": [],
        "entropy": [],
        "active_dim": [],
        "skewness": [],
        "kurtosis": [],
        "fractal": [],
        "position": []
    }

    for i in range(B):
        map_i = attn_maps_np[i]
        heatmap = map_i / (np.max(map_i) + 1e-6)
        binary_map = (heatmap > threshold).astype(np.uint8)

        # Intensity & variance
        intensity = np.mean(heatmap)
        variance = np.var(heatmap)

        # Entropy
        hist, _ = np.histogram(heatmap, bins=256, range=(0, 1))
        hist += 1e-6
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log(prob))

        # Activated area
        active_dim = np.sum(binary_map) / heatmap.size

        # Skewness, kurtosis
        skewness = skew(heatmap.flatten())
        kurtosis_c = kurtosis(heatmap.flatten())

        # Bounding box
        activated = np.argwhere(binary_map > 0)
        if activated.shape[0] > 0:
            y_min, x_min = activated.min(axis=0)
            y_max, x_max = activated.max(axis=0)
            x_min, x_max = x_min / side, x_max / side
            y_min, y_max = y_min / side, y_max / side
        else:
            x_min, x_max, y_min, y_max = 0, 0, 0, 0
        position = [x_min, x_max, y_min, y_max]

        # Fractal (Approx fallback): active_dim * (1 + kurtosis)
        fractal = active_dim * (1 + kurtosis_c)

        # Append to batch
        features["intensity"].append(intensity)
        features["variance"].append(variance)
        features["entropy"].append(entropy)
        features["active_dim"].append(active_dim)
        features["skewness"].append(skewness)
        features["kurtosis"].append(kurtosis_c)
        features["fractal"].append(fractal)
        features["position"].append(position)

    # Convert to tensors
    for key in features:
        if key == "position":
            features[key] = torch.tensor(features[key], dtype=torch.float32, device=device)  # [B, 4]
        else:
            features[key] = torch.tensor(features[key], dtype=torch.float32, device=device)  # [B]

    return features


# ============================= OTHER FUNCTIONS ============================= #

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


def _similarity_evaluation_single_node(node: dict, extracted_features: dict):
    """
    Compute cosine similarity between stored node features and current extracted ones.

    Args:
        node (dict): Graph node with 'features'.
        extracted_features (dict): Dict with keys matching stats_keys.
    """
    keys = ["intensity", "variance", "entropy", "active_dim", "skewness", "kurtosis", "fractal"]  # exclude 'position'
    stored_vec = torch.tensor([node["features"].get(k, 0.0) for k in keys], dtype=torch.float32)
    extracted_vec = torch.tensor([extracted_features.get(k, 0.0) for k in keys], dtype=torch.float32)

    sim = __cosine_similarity_torch(stored_vec, extracted_vec)
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


def __cosine_similarity_torch(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Cosine similarity between two 1D torch tensors, normalized to [0,1].
    """
    sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
    return (sim + 1.0) / 2.0


def update_graph_features(graph, extracted_features: dict, apply_similarity: bool = False):
    """
    Update all sign nodes in the graph using the new extracted features from one batch.

    Args:
        graph (dict): The graph with "nodes".
        extracted_features (dict): Dictionary of batched features (each key -> [B] or [B, 4]).
        apply_similarity (bool): Whether to compute similarity between stored and new features.

    Returns:
        dict: Updated graph.
    """
    batch_size = next(iter(extracted_features.values())).shape[0]

    for i in range(batch_size):
        sample_features = {
            k: v[i].item() if v.ndim == 1 else v[i].tolist()
            for k, v in extracted_features.items()
        }

        for node in graph["nodes"]:
            if node.get("type") != "sign":
                continue

            node.setdefault("count", 0)
            node.setdefault("features", {})

            for key, value in sample_features.items():
                if value is None:
                    continue

                if key not in node["features"] or node["count"] == 0:
                    node["features"][key] = value
                elif key == "position":
                    node["features"][key] = [
                        __update_weighted_mean(node["features"][key][j], value[j], node["count"])
                        for j in range(4)
                    ]
                else:
                    node["features"][key] = __update_weighted_mean(
                        node["features"][key], value, node["count"]
                    )

            if apply_similarity:
                _similarity_evaluation_single_node(node, sample_features)

            node["count"] += 1

    return graph