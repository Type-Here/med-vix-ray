from typing import Union

import torch

import torch.nn.functional as fc
from torch import Tensor

# ---- HELPER FUNCTIONS FOR FEATURE EXTRACTION ---- #

def _binarize_maps(heatmap: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Binarize the heatmap given a threshold.
    Args:
        heatmap: Tensor [H, W], values in [0,1]
        threshold: float threshold
    Returns:
        binary: FloatTensor [H, W] with 0/1 values
    """
    return (heatmap > threshold).float()

def _label_connected_components_kornia(binary: torch.Tensor, max_labels: int = 10) -> torch.Tensor:
        """
        GPU-accelerated connected components via Kornia.
        """
        import kornia.contrib as km
        # binary: [H, W], torch.float
        # add batch & channel dims
        comp = km.connected_components(binary.unsqueeze(0).unsqueeze(0).float())
        # comp: [1,1,H,W] with labels 0...N
        label_map = comp[0, 0].to(torch.int32)
        # optional: clamp to max_labels
        return torch.clamp(label_map, max=max_labels) # [0..max_labels]

def _label_connected_components(binary: torch.Tensor, max_labels: int = 10) -> torch.Tensor:
    """
    Label connected components in a binary map via iterative dilation (8-neighborhood).
    Args:
        binary: FloatTensor [H, W] of 0/1
        max_labels: safety cap on number of labels
    Returns:
        label_map: IntTensor [H, W] with region labels 1..n
    """
    # Ensure binary is a float tensor
    label_map = torch.zeros_like(binary, dtype=torch.int32)
    remaining = binary.clone()
    label_id = 1

    # Define a 3x3 kernel for dilation
    kernel = torch.ones((1, 1, 3, 3), device=binary.device)

    while remaining.sum() > 0 and label_id <= max_labels:
        seed = (remaining > 0).float()
        prev_seed = torch.zeros_like(seed)
        # Expand region until convergence
        while not torch.equal(seed, prev_seed):
            prev_seed = seed
            dilated = fc.conv2d(seed.unsqueeze(0).unsqueeze(0), kernel, padding=1)[0, 0]
            seed = (dilated > 0).float() * remaining
        label_map[seed.bool()] = label_id
        remaining = remaining * (1 - seed)
        label_id += 1

    return label_map # [H, W] with labels 0..n

# --- REGION STATS COMPUTATION --- #

def _compute_region_stats(heatmap: torch.Tensor, region_mask: torch.Tensor,
                          eps: float = 1e-6) -> tuple[Tensor, Union[int, float, bool]]:
    """
    Compute statistics for a single region mask.
    Args:
        heatmap: Tensor [H, W], normalized
        region_mask: FloatTensor [H, W] of 0/1 selecting region
        eps: small value to avoid division by zero
    Returns:
        stats: Tensor [F] with features
        area: int, number of pixels in the region
    """
    heads, wei = heatmap.shape

    # Apply mask to heatmap
    vals = heatmap[region_mask.bool()]
    # Compute area of the region
    area = region_mask.sum().item()

    # Basic stats
    mean_val = vals.mean()
    var_val = vals.var(unbiased=False)
    std_val = torch.sqrt(var_val + eps)

    # Entropy via histogram approximation
    hist_bins = 32
    hist = torch.histc(vals, bins=hist_bins, min=0.0, max=1.0) + eps
    prob = hist / hist.sum()
    entropy = -torch.sum(prob * torch.log(prob))

    # Skewness and kurtosis (moment-based)
    centered = vals - mean_val
    skewness = ((centered ** 3).mean()) / (std_val ** 3 + eps)
    kurtosis = ((centered ** 4).mean()) / (std_val ** 4 + eps)

    # Active area fraction
    active_dim = area / (heads * wei)

    # Fractal fallback
    fractal = active_dim * (1 + kurtosis)

    # Bounding box normalized
    coords = region_mask.nonzero(as_tuple=False).float()
    y_min, x_min = coords.min(dim=0)[0] / heads
    y_max, x_max = coords.max(dim=0)[0] / wei
    position = [x_min.item(), x_max.item(), y_min.item(), y_max.item()]


    # Create a single concatenated tensor with all features in order
    # returns Tensor[F] where F = 7 scalari + 4 position = 11
    return torch.cat([
          mean_val.unsqueeze(0),
          var_val.unsqueeze(0),
          entropy.unsqueeze(0),
          skewness.unsqueeze(0),
          kurtosis.unsqueeze(0),
          torch.tensor(active_dim, dtype=torch.float32, device=heatmap.device).unsqueeze(0),
          fractal.unsqueeze(0),
          torch.tensor(position, dtype=torch.float32, device=heatmap.device)
      ]), area  # Return both stats and area

# ================== EXTRACTION FEATURES FUNCTIONS ================= #

# Optimized feature extraction for multiregion attention maps
def extract_attention_batch_multiregion_torch(attn_maps: torch.Tensor, device: torch.device,
                                              threshold: float = 0.5, min_area_frac: float = 0.001,
                                              max_regions_per_image: int = 5) -> list[list[Tensor]]:
    """
    Optimized multiregion feature extraction with optional sign matching capability.

    Args:
        attn_maps (torch.Tensor): [B, 1, H, W] or [B, H, W]
        device (torch.device): computation device
        threshold (float): binarization threshold (or 'adaptive' for Otsu-like)
        min_area_frac (float): minimum region area fraction
        max_regions_per_image (int): max number of regions to extract per image

    Returns:
        list[dict]: list of dictionaries with region features
    """

    # Ensure correct shape and normalize batch at once
    if attn_maps.dim() == 4 and attn_maps.shape[1] == 1:
        attn_maps = attn_maps.squeeze(1)
    elif attn_maps.dim() != 3:
        raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {attn_maps.shape}")

    batch, head, wei = attn_maps.shape
    # features = {k: [] for k in ["intensity", "variance", "entropy", "skewness",
    #                            "kurtosis", "active_dim", "fractal", "position"]}

    # Batch normalize at once
    batch_min = attn_maps.view(batch, -1).min(dim=1)[0].view(batch, 1, 1)
    batch_max = attn_maps.view(batch, -1).max(dim=1)[0].view(batch, 1, 1)
    normalized_maps = (attn_maps - batch_min) / (batch_max - batch_min + 1e-6)

    # Use adaptive threshold if requested (Otsu-like: variance maximization)
    if threshold == 'adaptive':
        thresholds = []
        for i in range(batch):
            hist = torch.histc(normalized_maps[i], bins=256, min=0, max=1)
            cum_hist = torch.cumsum(hist, dim=0)
            total = cum_hist[-1]

            sum_total = torch.sum(torch.arange(256, device=device) * hist) / 256
            w_bg = cum_hist / total
            w_fg = 1 - w_bg

            # Avoid division by zero
            w_bg = torch.where(w_bg < 1e-6, torch.ones_like(w_bg), w_bg)
            w_fg = torch.where(w_fg < 1e-6, torch.ones_like(w_fg), w_fg)

            mean_bg = torch.cumsum(torch.arange(256, device=device) * hist, dim=0) / (256 * cum_hist + 1e-6)
            mean_fg = (sum_total - torch.cumsum(torch.arange(256, device=device) * hist, dim=0)) / (
                        256 * (total - cum_hist) + 1e-6)

            variance = w_bg * w_fg * (mean_bg - mean_fg) ** 2
            thresh_idx = torch.argmax(variance)
            thresholds.append((thresh_idx / 255).item())
    else:
        thresholds = [threshold] * batch

    return_features = []
    print_msg = True
    # Process each image
    for i in range(batch):
        heatmap = normalized_maps[i] # Normalized heatmap [H, W]
        binary = _binarize_maps(heatmap, thresholds[i]) # Binary map [H, W]
        try:
            labeled = _label_connected_components_kornia(binary, max_labels=10)  # Limit max regions
        except (RuntimeError, ModuleNotFoundError) as e:
            if print_msg:
                print(f"[WARN] Error in connected components kornia labeling: {e},"
                  f" trying np version.")
            print_msg = False
            labeled = _label_connected_components(binary, max_labels=10)


        # Extract stats for top regions by size
        regions = []
        for lbl in range(1, labeled.max().item() + 1):
            # Create a mask for the current region
            mask = (labeled == lbl).float()
            area = mask.sum().item()
            if area < min_area_frac * head * wei:
                continue

            # Here stats are computed for each region
            region_stats, area = _compute_region_stats(heatmap, mask)
            regions.append((area,region_stats))

        # Sort by area (descending) and take top K
        if len(regions) > max_regions_per_image:
            regions.sort(key=lambda x: x[0], reverse=True)
            regions = regions[:max_regions_per_image] # Keep only top K regions

        elif not regions:
            # Fallback to global stats
            # If no regions found, compute global stats
            mask = torch.ones_like(heatmap, dtype=torch.float32)
            stats, area = _compute_region_stats(heatmap, mask)
            # Wrap in a list to match the expected output format
            regions.append((area,stats))

        return_features.append([x for _, x in regions])  # Extract only the stats

    return return_features # list of B elements, each with a list of region features


# ============================= OTHER FUNCTIONS ============================= #

def _similarity_evaluation_single_node(node: dict, extracted_features_one_region: Tensor):
    """
    Compute cosine similarity between stored node features and current extracted ones.

    Args:
        node (dict): Graph node with 'features'.
        extracted_features_one_region (Tensor): Dict with keys matching stats_keys.

    Returns:
        float: Cosine similarity between stored and extracted features.
    """
    keys = ["intensity", "variance", "entropy", "active_dim", "skewness", "kurtosis", "fractal", "position"]  # exclude 'position'
    stored_vec = torch.tensor([node["features"].get(k, 0.0) for k in keys], dtype=torch.float32)

    return __cosine_similarity_torch(stored_vec, extracted_features_one_region)


def __cosine_similarity_torch(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Cosine similarity between two 1D torch tensors, normalized to [0,1].
    """
    sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
    return (sim + 1.0) / 2.0


# ---- UPDATE GRAPH USING MULTI-REGIO, MULTISIGNS EXTRACT FUNCTION ---- #

def find_match_and_update_graph_features(graph, extracted_features, device, stats_keys=None,
                                         update_features=True, is_inference=False):
    """
    Updates the graph nodes (type="sign") with feature statistics from matched regions.

    Args:
        graph (dict): Graph JSON containing "nodes".
        extracted_features (list[list[Tensor]]): List of lists of feature dictionaries.
        device (torch.device): Device for tensor operations.
        First list is over batch, second is over regions for each image.
        The tensor contains all stats features in tensors.
        stats_keys (list of str, optional): Keys to update. Defaults to a standard set.
        update_features (bool): If True, update the graph features of the most similar node.
        is_inference (bool): If True, return detected signs per sample.

    Returns:
        tuple: Updated graph and a dictionary of found signs (if is_inference=True, otherwise None).
    """
    stats_keys = stats_keys or ["intensity", "variance", "entropy", "active_dim",
                                "skewness", "kurtosis", "fractal", "position"]

    # Initialize signs_found with empty lists for all batch indices
    batch_size = len(extracted_features)
    signs_found = {i: [] for i in range(batch_size)}

    sign_vecs = []  # List to store sign vectors
    sign_ids = []  # List to store sign IDs
    sign_labels = []  # List to store sign labels
    for node in graph["nodes"]:
        if node.get("type") != "sign":
            continue

        # Set default values for node features
        node.setdefault("count", 0)

        vec = torch.tensor([node["features"].get(k, 0.0) for k in stats_keys if k != "position"],
                            device=device)
        # Add position as a separate feature
        pos = node["features"].get("position", [0.5, 0.5, 0.5, 0.5])
        pos_tensor = torch.tensor(pos, device=device)
        vec = torch.cat((vec, pos_tensor))

        sign_vecs.append(vec)
        sign_ids.append(node["id"])
        sign_labels.append(node["label"])

    # Stack the vectors to create a 2D tensor
    sign_vecs = torch.stack(sign_vecs)  # [N_signs, F]

    for i, regions_list in enumerate(extracted_features):
        for feature_tensor in regions_list:
            # Calculate cosine similarity all at once
            sims = fc.cosine_similarity(feature_tensor.unsqueeze(0), sign_vecs, dim=1)

            # Get the most similar sign
            most_similar_idx = torch.argmax(sims).item()
            matched_id = sign_ids[most_similar_idx]
            node = next(n for n in graph["nodes"] if n["id"] == matched_id)

            if update_features:  # In training mode
                __update_most_similar_node(node, feature_tensor, stats_keys)

            if is_inference:  # In inference mode, store the results
                if i not in signs_found:
                    signs_found[i] = []
                signs_found[i].append({
                    "id": sign_ids[most_similar_idx],
                    "similarity": sims[most_similar_idx].item(),
                    "label": sign_labels[most_similar_idx],
                    "stats": __from_tensor_to_dict(feature_tensor, stats_keys)
                })

    if is_inference:
        return graph, signs_found
    return graph, None

def __from_tensor_to_dict(tensor, keys):
    """
    Convert a tensor of clinical signs stats to a dictionary using the provided keys.
    Args:
        tensor (Tensor): The tensor to convert.
        keys (list of str): The keys to use for the dictionary.
    Returns:
        dict: The converted dictionary.
    """
    dict_result = {}
    for i, key in enumerate(keys):
        if key == "position":
            pos_start = len(keys) - 1  # last is 'position'
            dict_result["position"] = tensor[pos_start:].tolist()
            #dict_result[key] = tensor[-4:].tolist()
        else:
            dict_result[key] = tensor[i].item()
    return dict_result

def __update_most_similar_node(node, feature_tensor, stats_keys):
    """
    Update the most similar node with the new feature tensor.
    Tensor contains all stats features in order of stats_keys parameter.
    Position parameter is a list of 4 elements which are the last 4 elements of the tensor.
    It also updates the node count.
    Args:
        node (dict): The node to update.
        feature_tensor (Tensor): The new feature tensor.
        stats_keys (list of str): Keys to update in the node features.
    """
    # Update the node features with the new feature tensor
    for i, key in enumerate(stats_keys):
        if key == "position":
            # Handle position as 4-dim vector
            pos_values = feature_tensor[-4:].tolist()
            if key not in node["features"] or node["count"] == 0:
                node["features"][key] = pos_values
            else:
                node["features"][key] = [
                    (node["count"] * node["features"][key][j] + pos_values[j]) / (node["count"] + 1)
                    for j in range(4)
                ]
        else:
            new_val = feature_tensor[i].item()
            if key not in node["features"] or node["count"] == 0:
                node["features"][key] = new_val
            else:
                # Weighted moving average update
                prev = node["features"][key]
                count = node["count"]
                node["features"][key] = (count * prev + new_val) / (count + 1)

    node["count"] += 1