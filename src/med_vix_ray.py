import json
import math
from typing import Optional

import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as fc

import src.train_helpers
from settings import NUM_EPOCHS, LEARNING_RATE_TRANSFORMER, LEARNING_RATE_CLASSIFIER, UNBLOCKED_LEVELS, MIMIC_LABELS, \
    MODELS_DIR, LAMBDA_SIM, EPOCH_GRAPH_INTEGRATION, ALPHA_GRAPH, ATTENTION_MAP_THRESHOLD, \
    MIMIC_LABELS_MAP_TO_GRAPH_IDS, NER_GROUND_TRUTH, MANUAL_GRAPH, INJECT_BIAS_FROM_THIS_LAYER, EARLY_STOPPING_PATIENCE, \
    LEARNING_RATE_INPUT_LAYER, LAMBDA_KL
from src import general
from src.fine_tuned_model import SwinMIMICClassifier

import xai.attention_map_extractor as attention
import xai.feature_extract as xai_fe
import xai.edges_stats_update as update_edges
from src.train_helpers import CustomLRScheduler, EarlyStopper, focal_loss


def _compute_batch_features_vectors(features_dict, keys_order=None):
    """
    Convert a dictionary of batch features (each a tensor of shape [B] or [B, 4])
    into a single tensor of shape [B, f_dim].
    It expects the features to be in a dictionary format like:

    { "intensity": tensor of shape [B], ... }

    Args:
        features_dict (dict): Dictionary returned by extract_heatmap_features (for batch).
        keys_order (list, optional): List of keys specifying the order in which features should appear.
        Otherwise, the order is taken from the dictionary keys.

    Returns:
        torch.Tensor: Concatenated features tensor of shape [B, f_dim].
    """
    # Gather the feature values; if a key is missing, use 0.0 as default.
    # If no keys are provided, use the order of the features in the dictionary.
    keys = keys_order if keys_order is not None else list(features_dict.keys())
    feature_list = []
    for key in keys:
        # Ensure each feature tensor has shape [B, 1]
        tensor_val = (features_dict[key])
        if tensor_val.ndim == 1:  # reshape to [B, 1]
                      tensor_val = tensor_val.view(-1, 1)
        feature_list.append(tensor_val)

    # Optionally, add position features.
    if "position" in features_dict:
        pos = features_dict["position"].view(features_dict["position"].shape[0], -1)  # [B, 4]
        feature_list.append(pos)
    return torch.cat(feature_list, dim=1)


def _compute_feature_vector(features, keys=None):
    """
    Convert a dictionary of features into a tensor vector.
    This function allows for flexible feature selection and ordering.
    The features are expected to be in a dictionary format, where the keys are the feature names
    If no keys are provided, the function will use the order defined in self.stats_keys
    by capturing the first sign node order.

    Args:
        features (dict): Dictionary of features (e.g., {"intensity": ..., "variance": ..., "entropy": ..., ...}).
        keys (list, optional): List of keys specifying the order in which features should appear.
                               If None, a default order is used, following the order of the keys in the dictionary.

    Returns:
        torch.Tensor: A 1D tensor containing the selected features.
    """
    if keys is not None:
        # Gather the feature values; if a key is missing, use 0.0 as default.
        feature_list = [features.get(k, 0.0) for k in keys]
    else:
        # If no keys are provided, use the order of the features in the dictionary.
        feature_list = [features.get(k, 0.0) for k in features.keys()]

    # Convert the list to a tensor. Optionally, you can set the data type.
    return torch.tensor(feature_list, dtype=torch.float32)


def build_adjacency_matrix(graph_json, num_diseases, num_signs, scale_corr=1.0, scale_find=0.7):
    # Offset for sign node IDs in the graph
    delta_disease_sign_ids = 7
    print(f"Building adjacency matrix for {num_diseases} diseases and {num_signs} signs.")

    # Initialize the blocks with float32 type.
    matr_dd = np.zeros((num_diseases, num_diseases), dtype=np.float32)  # disease-disease
    matr_ds = np.zeros((num_diseases, num_signs), dtype=np.float32)  # disease-sign

    # Process each edge in the JSON
    for edge in graph_json["links"]:
        weight = edge.get("weight", 1.0)
        if edge["relation"] == "correlation":
            i = int(edge["source"])  # disease index
            j = int(edge["target"])  # disease index
            # Scale correlation edges
            matr_dd[i, j] = weight * scale_corr
            matr_dd[j, i] = weight * scale_corr  # if undirected
        elif edge["relation"] == "finding":
            # Assume the source is a disease and target is a sign node.
            i = int(edge["source"])  # disease index
            k = int(edge["target"]) - delta_disease_sign_ids - num_signs  # sign index
            matr_ds[i, k] = weight * scale_find

    matr_sd = matr_ds.T  # sign-to-disease (transpose)
    matr_ss = np.zeros((num_signs, num_signs), dtype=np.float32)  # no sign-to-sign edges

    # Create full block matrix: rows/cols: [diseases; signs]
    #A_full = np.block([[mA_dd, mA_ds],
    #                   [mA_sd, mA_ss]])

    # Instead of using np.block, concatenate the sub-blocks:
    top = np.concatenate((matr_dd, matr_ds), axis=1)
    bottom = np.concatenate((matr_sd, matr_ss), axis=1)
    matrix_full = np.concatenate((top, bottom), axis=0)

    return matrix_full


# ========================= GRAPH BIAS ADAPTER CONV =========================


class GraphBiasAdapterConv(nn.Module):
    def __init__(self, hidden_channels=8):
        """
        Convolutions to adapt the graph matrix G (NxN) to a shape compatible with attention.
        Params:
            hidden_channels (int): Number of hidden channels for the convolutional layers.
        """
        super(GraphBiasAdapterConv, self).__init__()

        # Convolutional layers to adapt the graph matrix to the attention scores.
        self.adapter = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the Conv using Xavier (Glorot) uniform initialization.
        This is a common practice for initializing weights in neural networks
        in order to avoid vanishing/exploding gradients.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, g_matrix):  # G shape: [B, N, N] or [B, H, N, N]
        if g_matrix.dim() == 3:
            g_matrix = g_matrix.unsqueeze(1)  # Add channel: [B, 1, N, N]
        g_out = self.adapter(g_matrix)  # Apply conv
        return g_out.squeeze(1)  # Remove Channel: [B, N, N]

# ========================= GRAPH ATTENTION BIAS =========================


class GraphAttentionBias(nn.Module):
    def __init__(self, alpha=ALPHA_GRAPH, conv=None, d_k=64, num_injected_layers=2, total_layers=4):
        """
        Initialize the graph attention bias module.
        This module modifies the attention scores injecting the graph information
        Parameters:
            alpha (float): Scaling factor for the graph matrix. Default is taken from settings.py.
            conv (list): List of GraphBiasAdapterConv Network (convolutional for now) to adapt the graph matrix,
            1 for each layer injected.
            to the attention scores. Note: it is needed!
            d_k (int): Dimension for the key/query/value embeddings. Default is 64.
            num_injected_layers (int): Number of layers to inject the graph bias. Default is 2.
            total_layers (int): Total number of layers in the model. Default is 4.
        Raises:
            ValueError: If conv parameter is left to None.
        """
        super(GraphAttentionBias, self).__init__()
        # Minimum alpha value for graph attention bias
        self.min_alpha = 0.05
        eps = 1e-6
        ratio = max((alpha - self.min_alpha) / (1 - self.min_alpha + eps), eps)
        w_init = np.log(ratio / (1 - ratio + eps))

        # Alpha logits for each layer where the graph bias is injected.
        self._alpha_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(w_init, dtype=torch.float32))
            for _ in range(num_injected_layers)
        ])
        #self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))  # or fix as a hyperparameter

        if conv is None:
            raise ValueError("Conv value needed!")
        self.conv = conv
        self.d_k = d_k  # Dimension for the key/query/value embeddings
        self._printed_debug = False  # Flag to control debug printing

        self._num_injected_layers = num_injected_layers  # Number of layers to inject the graph bias
        self._total_layers = total_layers  # Total number of layers in the model

        # Considering Layers numbered from 0 to total_layers - 1
        self._first_layer_injected = total_layers - num_injected_layers  # First layer to inject the graph bias

    def get_alpha(self, layer_idx):
        """
        Compute learnable alpha in [min_alpha, 1] for a given layer.
        Args:
            layer_idx (int): Index of the layer.
        Returns:
            torch.Tensor: Computed alpha value for the layer.
        """
        alpha_idx = layer_idx - self._first_layer_injected
        logit = self._alpha_logits[alpha_idx]
        return self.min_alpha + (1 - self.min_alpha) * torch.sigmoid(logit)

    def forward(self, attn_scores, graph_adj_matrix, layer_idx):

        # If the layer index is less than the first injected layer, return the original attention scores.
        if layer_idx < self._first_layer_injected:
            return attn_scores

        # If attn_scores is 3D, add a dimension to make it 4D
        if attn_scores.dim() == 3:
            attn_scores = attn_scores.unsqueeze(1)  # Now [B, 1, N, N]

        # attn_scores: [B, H, N, M], graph_adj_matrix: [N, M]
        ba, he, n_q, n_k = attn_scores.shape # a

        # Add batch dimension to the graph adjacency matrix if needed
        if graph_adj_matrix.dim() == 2:
            graph_adj_matrix = graph_adj_matrix.unsqueeze(0)  # → [1, N0, N0]

        # Expand the graph adjacency matrix to match the attention scores shape
        if graph_adj_matrix.dim() == 3:  # [1, N0, N0]
            g_expanded = graph_adj_matrix.unsqueeze(1)  # → [1, 1, N0, N0]
        else:
            g_expanded = graph_adj_matrix

        g_resized = torch.nn.functional.interpolate(g_expanded, size=(n_q, n_k),
                                                    mode='bilinear',align_corners=False)
        # Replicates the resized matrix across the batch dimension (ba),
        # resulting in a tensor of shape [B, 1, N, N]:
        g_resized = g_resized.expand(ba, -1, -1, -1)  # [B, 1, N, N]

        # Call the convolutional layer to adapt the graph matrix to the attention scores:
        # Normalize with tanh
        conv_idx = layer_idx - self._first_layer_injected

        g_adapted = torch.tanh(self.conv[conv_idx](g_resized))
        #g_adapted = self.conv(g_resized) # [B, N, N]

        # Debugging information only first time
        if not self._printed_debug:
            print(f"[GraphAttentionBias] attn_scores: {attn_scores.shape}")
            print(f"[GraphAttentionBias] G_resized: {g_resized.shape} - Pre conv")
            print(f"[GraphAttentionBias] G_adapted: {g_adapted.shape} - Post conv")
            self._printed_debug = True

        # Standard scaled dot-product attention factor:
        # scaling = np.sqrt(self.d_k)

        # Obtain the alpha value for the current layer
        alpha = self.get_alpha(layer_idx)

        # Modify the attention scores by adding the adapted graph matrix:
        # No scaling needed in this case as swin transformer already scales
        # the attention scores using cosine similarity and tau division
        modified_scores = attn_scores + alpha * g_adapted.unsqueeze(1)  # [B, 1, N, N]

        # modified_scores = attn_scores / (attn_scores.shape[-1]**0.5) + self.alpha * g_resized
        return torch.softmax(modified_scores, dim=-1)


# ========================= GRAPH ATTENTION MODULE =========================


class GraphAttentionModule(nn.Module):
    def __init__(self, num_classes, d_k, graph_matrix=None, ner_ground_truth=None):
        """
        Initialize the graph attention module.

        Args:
            num_classes (int): Number of classes (or graph nodes).
            d_k (int): Dimension for the key/query/value embeddings.
            graph_matrix (np.array or torch.Tensor, optional): The pre-computed graph matrix (shape: num_classes x num_classes).
                If None, defaults to a zero matrix.
            ner_ground_truth (dict): Ground truth data for NER. This should be a dictionary
        """
        super(GraphAttentionModule, self).__init__()
        self.num_classes = num_classes
        self.d_k = d_k

        # Placeholder for ground truth data
        self.ner_ground_truth = ner_ground_truth

        # Learnable projection for each class.
        self.W_e = nn.Parameter(torch.randn(num_classes, d_k))
        # Output projection from the aggregated embedding to a scalar bias per class.
        self.W_out = nn.Linear(d_k, 1)

        # Register the graph matrix as a buffer.
        if graph_matrix is not None:
            if not torch.is_tensor(graph_matrix):
                graph_matrix = torch.tensor(graph_matrix, dtype=torch.float32)
            self.register_buffer("G", graph_matrix)
        else:
            self.register_buffer("G", torch.zeros(num_classes, num_classes))

    def forward(self, base_logits):
        """
        Compute the graph bias.

        Args:
            base_logits (torch.Tensor): Base logits from the model (shape: [B, num_classes]).
        Returns:
            torch.Tensor: A bias term of shape [B, num_classes] to be added to the base logits.
        """
        # Compute image-conditioned embeddings for each class.
        # e_img has shape (B, C, d_k)
        e_img = base_logits.unsqueeze(-1) * self.W_e.unsqueeze(0)
        scaling = np.sqrt(self.d_k)
        # Compute dot-product attention scores, shape: (B, C, C)
        score = torch.bmm(e_img, e_img.transpose(1, 2)) / scaling
        # Add the graph matrix (broadcast along batch dimension)
        score = score + self.G.unsqueeze(0)
        # Apply softmax to obtain attention weights.
        atten = torch.softmax(score, dim=-1)
        # Aggregate the embeddings according to the attention weights.
        aggr = torch.bmm(atten, e_img)
        # Project each aggregated embedding to a scalar bias.
        bias = self.W_out(aggr).squeeze(-1)
        return bias

    def update_edge_weights_with_ground_truth(self, graph_json, study_id, gt_labels):
        """
        Update the weights of 'finding' edges in the graph using ground truth information from reports.

        For each finding edge (connecting a disease node to a sign node), update its weight based on:
          - If the ground truth for the current study indicates the sign is present (polarity = 1),
            use the ground truth similarity value s_gt.
          - If the ground truth indicates a negative finding (polarity = 0) or the sign is not mentioned,
            update the weight toward 0.

        The new weight is computed using a function in xai.edges_stats_update.py module.
        The function used at the moment is bayes_b_distribution, which uses a Beta distribution:
        E[weight] = α / (α + β)
        where:
          - α = sum of s_gt for the number of positive observations
          - β = sum of s_gt for the number of negative observations
          - s_gt is the similarity score between the report word and the sign node label (or synonym)

        Args:
            graph_json (dict): Graph data with "nodes" and "edges". Each sign node must have "id", "type"=="sign",
                               and stored "features" (a dict).
            study_id (str): The identifier for the current study, used to look up ground truth.
            gt_labels (list): Each position is the value of the node for a specific xr, 1.0 if present, 0.0 if not present.
                The order of the labels is defined by the order of MIMIC_LABELS list in settings.py.

        Returns:
            dict: Updated graph_json with modified weights for finding edges.
        """
        # Convert extracted features to a feature vector (using your helper function).
        # Not used but could be an option.
        # new_feat_vec = _compute_feature_vector(extracted_features, keys_order)  # numpy array

        # Look up ground truth for the current study.
        # Assume ground_truth is available (e.g., a global variable) with structure:
        # { study_id:
        #   { "sign_node_id": [s_gt, polarity],
        #       ...,
        #     "dicom_ids": [dicom_id1, dicom_id2, ...],
        #   }, ...
        # }
        gt_entry = self.ner_ground_truth.get(study_id, {})  # Returns a dict for this study.
        positive_labels = [] # List with node IDs of positive labels

        # gt_labels is a list of 0.0 or 1.0 from ground truth
        for i, gt_label in enumerate(gt_labels):
            if gt_label > 0.0:
                # If label is positive, get its position in the labels list.
                label_name = MIMIC_LABELS[i]
                positive_labels.append(MIMIC_LABELS_MAP_TO_GRAPH_IDS[label_name])

        return update_edges.bayes_b_distribution(graph_json, positive_labels, gt_entry)


# ========================= GRAPH NUDGER =========================


class GraphNudger(nn.Module):
    def __init__(self, eta=0.01):
        super(GraphNudger, self).__init__()
        self.eta = eta  # Nudging learning rate
        self.sign_to_diseases = None  # Dictionary to store sign-to-disease mappings

    def forward(self, device, signs_found, graph,
                num_diseases, grad_output_batch):
        """
        Compute a nudging bias vector for each sample in the batch based on the difference
        between the extracted heatmap features and the stored features in the graph's sign nodes.

        For each sample and for each disease node d, we sum over all finding edges from d to sign nodes:

            Δb_d^(i) = η * ∑_{edge: source=d, type='finding'} (w_{d,s} * sim(f_att^(i), f_s) * g^(i))

        where:
          - f_att^(i) is the heatmap sign related to sample i,
          - f_s is the stored sign node,
          - sim(·,·) is the cosine similarity (normalized to [0,1]),
          - g^(i) is the gradient vector for sample i (elementwise used),
          - w_{d,s} is the edge weight.

        Args:
            device (torch.device): device to use for computations (CPU or GPU).
            signs_found (dict): dictionary with sign node IDs as keys and their features as values.
            graph (dict): the full graph (nodes + links)
            num_diseases (int): number of disease classes
            grad_output_batch (torch.Tensor): [B, f_dim] classifier gradient output

        Returns:
            torch.Tensor: [B, num_diseases] nudging bias
        """
        from collections import defaultdict

        batch = grad_output_batch.size(0)
        nudges = torch.zeros(batch, num_diseases, device=device)

        self.sign_to_diseases = defaultdict(list)

        for edge in graph["links"]:
            if edge["relation"] == "finding":
                d = int(edge["source"])
                s = int(edge["target"])
                w = edge.get("weight", 1.0)
                self.sign_to_diseases[s].append((d, w))

        # Get the Gradient for the current sample
        grad_norms = torch.norm(grad_output_batch, dim=1)  # [B]

        # Process each sample
        for i in range(batch):
            g = grad_norms[i].item()
            list_of_signs = signs_found[i] # Get list of dicts containing sign node IDs and their features
            for s_dict in list_of_signs:
                sim = s_dict.get("similarity", 0)
                s_id = int(s_dict["id"])
                for d, w in self.sign_to_diseases.get(s_id, []):
                    nudges[i, d] += self.eta * w * sim * g

        return nudges


# ========================= SWIN MIMIC + GRAPH CLASSIFIER =========================


class SwinMIMICGraphClassifier(SwinMIMICClassifier):
    """
    This subclass extends SwinMIMICClassifier by integrating graph-based information.
    It uses an AttentionMap to extract heatmap features during training/inference,
    updates a clinical findings graph based on those features,
    and—after a given number of epochs—incorporates the graph information into the logits.
    """
    def __init__(self, num_classes=len(MIMIC_LABELS), graph_json=None,
                 graph_integration_start_epoch=EPOCH_GRAPH_INTEGRATION,
                 ner_ground_truth_path=NER_GROUND_TRUTH, device = None):
        """
        Args:
            num_classes (int): Number of classes.
            graph_json (dict): JSON structure holding your graph. If None, an empty graph is created.
            graph_integration_start_epoch (int): After which epoch to incorporate graph info.
            device(torch.device): set torch device for cpu or cuda computations. If left to none defaults to 'cpu'
            ner_ground_truth_path (str): Path to the ground truth file for NER.
            If None defaults to swin_model.patch_embed.num_patches

        """
        super(SwinMIMICGraphClassifier, self).__init__(num_classes=num_classes)

        # Flag for inference mode
        self.signs_found = None
        self.is_inference = False

        # Set create optimizer to custom function
        self._create_optimizer = src.train_helpers.create_optimizer
        # Set placeholder for learning rate scheduler and classifier
        self._lr_scheduler = None
        self._early_stopper = None

        # Set number of output labels
        self.num_classes = num_classes
        # Set device from init (cpu or cuda)
        self.device = device if device is not None else torch.device("cpu")

        # Placeholder for base logits, to be used for graph bias computation.
        self.base_logits = None
        # Placeholder for the classifier logits, to be used for loss computation.
        self.classifier_logits = None
        # Placeholder for the classifier gradient, to be used for nudging backpropagation.
        self.classifier_grad = None
        # Placeholder for graph features list to be used for nudging. Will be filled on first call needed.
        self.stats_keys = None
        # Placeholder for the features extracted from the attention maps.
        self.features_dict_batch = None  # This will be a dictionary of features for each sample in the batch.

        # Placeholder for log of features extracted and signs found number in debug mode.
        self._debug_features_log = None
        self.save_debug_features_log = False  # Flag to control whether to save debug features log.

        # Flag indicating whether graph guidance has been injected into transformer layers.
        self.is_graph_used = False
        # Flag for Graph Nudger mode.
        self.is_using_nudger = False

        # Epoch threshold to activate graph-based mechanisms (both transformer bias & final nudging).
        self.graph_integration_start_epoch = graph_integration_start_epoch
        # Current epoch counter (to be updated during training).
        self.current_epoch = -1 # Set to -1 to indicate no training yet.
        # Total epochs; useful for calculating activation timing.
        self.total_epochs = NUM_EPOCHS
        # ner_ground_truth: save as variable to be used in the graph attention bias module.
        # This is a dictionary with the ground truth for each study_id.
        if ner_ground_truth_path is None:
            raise ValueError("ner_ground_truth cannot be None.")
        with open(ner_ground_truth_path, 'r') as nerf:
            self.ner_ground_truth = json.load(nerf)

        # Initialize graph information.
        # Expecting graph_json to contain "nodes" and "edges". For our vocabulary:
        #   - Sign nodes (clinical findings) and label nodes (MIMIC pathologies) are in "nodes".
        #   - Finding edges: label-to-sign; Correlation edges: label-to-label.
        if graph_json is None:
            self.graph = {"nodes": [], "edges": []}
        else:
            self.graph = graph_json
            # Calculate the number of sign nodes: total nodes minus number of label nodes.
            num_signs = len(graph_json["nodes"]) - len(MIMIC_LABELS)
            # Build the full adjacency matrix (using our helper function).
            graph_matrix = build_adjacency_matrix(graph_json, num_diseases=len(MIMIC_LABELS), num_signs=num_signs)
            # Save the computed adjacency matrix into the graph dictionary for later use.
            self.graph_matrix = graph_matrix

        # Register the forward hook on the target attention module.
        self.__patch_window_attention_hook()

        # Initialize the GraphAttentionBias module
        # which will be used to inject the graph bias into the attention scores.

        # Get the dimension of the model and number of heads from the first block of the first layer.
        print("Initializing graph attention bias...")
        d_model = self.swin_model.layers[0].blocks[0].attn.dim
        num_heads = self.swin_model.layers[0].blocks[0].attn.num_heads
        num_layers = len(self.swin_model.layers)  # Total number of layers in the model

        # Number of layers to inject bias into.
        bias_inj_num_layers = num_layers - INJECT_BIAS_FROM_THIS_LAYER

        # Init the graph bias adapter.
        # This Conv Layers will adapt the graph matrix to the attention scores.
        print(" - Initializing graph bias adapter...")
        self.conv_adapter = [GraphBiasAdapterConv() for _ in range(bias_inj_num_layers)]

        # Calculate the dimension for each head.
        d_k = d_model // num_heads
        # Initialize the graph bias module.
        self.graph_bias_module = GraphAttentionBias(alpha=ALPHA_GRAPH, conv=self.conv_adapter,
                                                    d_k=d_k, num_injected_layers=bias_inj_num_layers,
                                                    total_layers=num_layers)

        print("Initializing attention map modules...")
        # Initialize the AttentionMap module.
        # This module will extract attention maps from the second last layer (before the classifier head)
        # using techniques such as rollout-like attention to later compare with sign node statistics.
        self.attention_map_generator = attention.SelfAttentionMapExtractor(model=self.swin_model)

        # Initialize the GraphAttentionModule.
        # This module uses the (normalized) adjacency matrix (from both correlation and finding edges)
        # to compute a bias that will be injected into transformer layers.
        self.graph_attention_module = GraphAttentionModule(num_classes=num_classes, d_k=d_k,
                                                           graph_matrix=self.graph_matrix,
                                                           ner_ground_truth=self.ner_ground_truth)

        # Initialize the GraphNudger module.
        # This module is responsible for the final nudging operation on the classifier head:
        # it compares attention-derived features with stored sign node statistics and computes a weight update.
        print("Initializing graph nudger...")
        self.graph_nudger = GraphNudger(eta=0.01)  # nudging learning rate

        # Note: self.classifier is already defined in the parent class (SwinMIMICClassifier).

    def __patch_window_attention_hook(self):
        """
        Replaces the forward method of the last WindowAttention module to store attention weights.
        """
        attn_module = self.swin_model.layers[-2].blocks[-1].attn  # Last attention module

        def new_forward(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
                From the original forward, we modify it to compute and hook attention scores
                "attn_weights" is the attribute where we save the attention weights.
                See: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py
            """
            # Call the original forward (which now includes attention computation)
            bat_, n_tok, embed_dim = x.shape

            if attn_module.q_bias is None:
                qkv = attn_module.qkv(x)
            else:
                qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
                if attn_module.qkv_bias_separate:
                    qkv = attn_module.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = fc.linear(x, weight=attn_module.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(bat_, n_tok, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # Cosine attention
            attn = (fc.normalize(q, dim=-1) @ fc.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(attn_module.logit_scale, max=math.log(1. / 0.01)).exp()
            attn = attn * logit_scale

            relative_position_bias_table = attn_module.cpb_mlp(attn_module.relative_coords_table).view(-1,
                                                                                                       attn_module.num_heads)
            relative_position_bias = relative_position_bias_table[attn_module.relative_position_index.view(-1)].view(
                attn_module.window_size[0] * attn_module.window_size[1],
                attn_module.window_size[0] * attn_module.window_size[1],
                -1
            )  # [Wh*Ww, Wh*Ww, nH]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, attn_module.num_heads, n_tok, n_tok) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, attn_module.num_heads, n_tok, n_tok)
                attn = attn_module.softmax(attn)
            else:
                attn = attn_module.softmax(attn)

            # Save the attention weights!
            attn_module.attn_weights = attn.detach() # HOOK

            attn = attn_module.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(bat_, n_tok, embed_dim)
            x = attn_module.proj(x)
            x = attn_module.proj_drop(x)
            return x

        # Inject the new forward
        attn_module.forward = new_forward

    def compute_graph_bias(self, base_logits):
        """
        Compute a refined graph bias using the GraphAttentionModule.
        """
        return self.graph_attention_module(base_logits)

    def __compute_feature_vector(self, features, keys=None):
        """
        Convert a dictionary of features into a tensor vector.
        This function allows for flexible feature selection and ordering.
        The features are expected to be in a dictionary format, where the keys are the feature names
        If no keys are provided, the function will use the order defined in self.stats_keys
        by capturing the first sign node order.
        Args:
            features (dict): Dictionary of features (e.g., {"intensity": ..., "variance": ..., "entropy": ..., ...}).
            keys (list, optional): List of keys specifying the order in which features should appear.
                                   If None, a default order is used.

        Returns:
            torch.Tensor: A 1D tensor containing the selected features.
        """
        if keys is None and self.stats_keys is None:
            # Defining a default order from graph order
            # Get a sign node from the graph
            sign_node = next((node for node in self.graph["nodes"] if node["type"] == "sign"), None)
            feat = [feat for feat in sign_node["features"]]
            self.stats_keys = feat

            # If features is a single dictionary, compute single feature vector.
            #return _compute_feature_vector(features, keys=self.stats_keys)

            # If features is a batch, compute batch feature vectors.
        return _compute_batch_features_vectors(features, keys_order=self.stats_keys)

    def __inject_graph_bias_in_transformer(self, graph_adj_matrix, use_graph_guidance):
        """
        For each transformer block in the model, override its attention calculation.
        This is where we inject the graph bias into the attention scores.
        This is done by modifying the forward function of the attention module.

        Args:
            graph_adj_matrix (np.array): The adjacency matrix of the graph.
            use_graph_guidance (bool): Whether to use graph guidance.
        """
        # For example, inject bias only in layers with index >= threshold_layer.
        threshold_layer = INJECT_BIAS_FROM_THIS_LAYER  # TODO: adjust based on training results

        if not use_graph_guidance:
            return  # do nothing

        # Ensure graph_adj_matrix is a torch tensor:
        if not torch.is_tensor(graph_adj_matrix):
            graph_adj_matrix = torch.tensor(graph_adj_matrix, dtype=torch.float32,
                                            device=self.swin_model.patch_embed.proj.weight.device)

        #graph_bias_module = GraphAttentionBias(alpha=ALPHA_GRAPH)
        # Assume self.swin_model.layers is a list of layers, each with blocks that have an "attn" module.
        for layer_idx, layer in enumerate(self.swin_model.layers):
            # Only inject bias in layers >= threshold_layer.
            if layer_idx < threshold_layer:
                continue
            for block_idx, block in enumerate(layer.blocks):
                original_forward = block.attn.forward

                layer_index_in_bias = threshold_layer - layer_idx

                def new_forward(x, *args, orig_forward=original_forward, layer_index=layer_index_in_bias, **kwargs):
                    # Get raw attention scores (this requires that the original attn returns them)
                    attn_scores = orig_forward(x, *args, **kwargs)
                    # Inject graph bias using our module. Expecting attn_scores to be of shape [B, H, N, N]

                    modified_attn = self.graph_bias_module(attn_scores, graph_adj_matrix, layer_index)
                    return modified_attn

                block.attn.forward = new_forward
                print(f"Injected graph bias into layer {layer_idx}, block {block_idx}")

    def forward(self, x, use_graph_guidance=True, study_ids=None):
        """
        Perform a forward pass through the model with optional graph-guided nudging.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].
            use_graph_guidance (bool): Whether to apply graph-guided nudging.
            study_ids (list, optional): List of study IDs for the current batch. Used for ground truth lookups.

        Returns:
            torch.Tensor: Final logits tensor of shape [B, num_classes].

        Workflow:
            1. Inject graph bias into attention layers once if enabled.
            2. Extract base logits from the backbone and classifier head.
            3. During training:
                - Extract attention-based heatmap features.
                - Update graph sign nodes with extracted features and similarity scores.
            4. If graph guidance is disabled, return raw classifier logits.
            5. If nudging is active:
                - Compute a bias vector based on the similarity between extracted features and sign node features.
                - Add the bias vector to the classifier logits.
            6. Return the adjusted or base logits.

        Note:
            - self.is_using_nudger (bool): If True, nudges the classifier head using the GraphNudger module
            (uses the attention map and stats feature in sign nodes).
            Set the flag accordingly in the training loop.
        """
        self.classifier_grad = None  # Reset the classifier gradient for each forward pass.
        self.signs_found = None  # Reset the signs found for each forward pass.
        self.features_dict_batch = None  # Reset the features dictionary for each forward pass.

        # 1b. Else, if graph guidance is active, use graph
        if use_graph_guidance and not self.is_graph_used:
            # Inject graph bias into the transformer layers if not already done.
            # This is done only once, at the beginning of training so flag is set to True.
            self.__inject_graph_bias_in_transformer(self.graph_matrix, use_graph_guidance)
            self.is_graph_used = True

        # 2. Call the forward method of the parent class.
        base_logits = super(SwinMIMICGraphClassifier, self).forward(x, use_classifier=False)
        self.base_logits = base_logits  # store for potential use

        # 3. Compute classifier logits.
        self.classifier_logits = self.classifier(base_logits)  # shape: [B, num_classes]

        # 3b. Register the hook on classifier_logits to capture its gradient during backpropagation.
        if self.training:
            self.classifier_logits.register_hook(self._save_classifier_grad)

        # 4. Extract attention maps and features
        att_maps_batch = self.attention_map_generator.extract(x)
        features_dict_batch = xai_fe.extract_attention_batch_multiregion_torch(att_maps_batch, self.device,
                                                    threshold=ATTENTION_MAP_THRESHOLD, # Now defaults to 'percentile' threshold
                                                    current_epoch=self.current_epoch)

        self.features_dict_batch = features_dict_batch  # Store for use in training loop

        # 4b. Update graph from extracted features
        # Update the graph with the new feature statistics.
        _ , signs_found = xai_fe.find_match_and_update_graph_features(
            self.graph,
            extracted_features=features_dict_batch,
            device=self.device,
            update_features=self.training,
            is_inference=self.is_inference or use_graph_guidance,
            use_softmax=True,
            softmax_params={'temperature': 0.5 if self.training and self.current_epoch < 4 else 0.2,
                            'top_k': 2,
                            'use_reports': self.training and self.current_epoch < 2,
                            'study_ids': study_ids
                            }
        )

        self.signs_found = signs_found

        # If debugging is enabled, update the number of signs found and features extracted
        if self.save_debug_features_log:
            # Make sure the current epoch entry exists in the debug log
            if self.current_epoch not in self._debug_features_log:
                self._debug_features_log[self.current_epoch] = {"signs_found": 0, "features_extracted": 0}

            # Update the counts directly without creating a new dictionary
            self._debug_features_log[self.current_epoch]["signs_found"] += len(signs_found)
            self._debug_features_log[self.current_epoch]["features_extracted"] += len(features_dict_batch)

        # 1a. If graph guidance is not active return the classifier logits directly.
        if not use_graph_guidance:
            return self.classifier_logits


        # 5. Compute the graph bias using the Nudger module.
        # Use the attention map features and stats features.
        if self.is_using_nudger:
            grad_output_batch = (
                self.classifier_grad if self.classifier_grad is not None
                else torch.ones_like(self.classifier_logits)
            )

            update_vector = self.graph_nudger(
                device=self.device,
                signs_found=signs_found,
                graph=self.graph,
                num_diseases=len(MIMIC_LABELS),
                grad_output_batch=grad_output_batch
            )
            # Transfer the update vector to the same device as the classifier logits.
            update_vector = update_vector.to(self.device)

            # 6a. Then final logits become:
            final_logits = self.classifier_logits + update_vector  # where classifier_logits is [B, num_diseases]
        else:
            # 6a.2 If nudging is not used, we can still compute the graph bias.
            final_logits = self.classifier_logits

        # 7. Return the final logits.
        return final_logits

    def _save_classifier_grad(self, grad):
        """
        Hook function to capture the gradient of the classifier output.
        This function stores the gradient in self.classifier_grad.
        """
        self.classifier_grad = grad.detach()

    def train_model(self, train_loader, num_epochs=NUM_EPOCHS,
                    learning_rate_swin=LEARNING_RATE_TRANSFORMER,
                    learning_rate_classifier=LEARNING_RATE_CLASSIFIER,
                    learning_rate_input=LEARNING_RATE_INPUT_LAYER,
                    layers_to_unblock=UNBLOCKED_LEVELS,
                    loss_fn_param=focal_loss,
                    optimizer_param=None,
                    lambda_sim=LAMBDA_SIM,
                    lambda_kl=LAMBDA_KL,
                    patience=EARLY_STOPPING_PATIENCE, use_validation=True,
                    validation_loader=None,
                    debug_features_log=False):
        """
        Custom training loop that incorporates the graph-based loss regularization.

        For each batch:
          - Compute adjusted logits using the forward pass (which applies graph guidance).
          - Retrieve the base classifier logits (without graph bias) from self.base_logits.
          - Compute the graph bias as: graph_bias = adjusted_logits - classifier_logits.
          - Compute classification loss (e.g. BCEWithLogitsLoss) on the adjusted logits.
          - Add a regularization term: λ * mean( graph_bias² ).

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            num_epochs (int): Number of epochs to train.
            learning_rate_swin (float): Learning rate for the transformer blocks.
            learning_rate_classifier (float): Learning rate for the classifier head.
            learning_rate_input (float): Learning rate for the input layer.
            layers_to_unblock (int): Number of layers to unblock in the Swin Transformer.
            optimizer_param (torch.optim.Optimizer, optional): Optimizer for training. If None, defaults to AdamW.
            loss_fn_param (callable, optional): Loss function. If None, defaults to BCEWithLogitsLoss.
            lambda_sim (float): Regularization parameter for similarity loss.
            lambda_kl (float): Regularization parameter for KL divergence loss.
            patience (int): Number of epochs for early stopping.
            use_validation (bool): Whether to use validation data for early stopping.
            validation_loader (DataLoader, optional): DataLoader for the validation dataset.
            debug_features_log (bool): If True, log the number of features extracted and signs found per epoch.
        """
        if debug_features_log:
            self.save_debug_features_log = True
            self._debug_features_log = {}

        ## Set the model to training mode.
        self.train()
        # Set the GraphNudger Module to be used.
        # self.is_using_nudger = True
        # Unfreeze the specified layers in the transformer.
        self._unblock_layers(layers_to_unblock)

        # Define optimizer with parameter groups.
        optimizer = self._create_optimizer(self, layers_to_unblock,
                                           learning_rate_input=learning_rate_input,
                                           learning_rate_swin=learning_rate_swin,
                                           learning_rate_classifier=learning_rate_classifier,
                                           optimizer_param=optimizer_param)

        # Attach Custom LR Scheduler
        self._lr_scheduler = CustomLRScheduler(optimizer)

        # Attach EarlyStopping
        self._early_stopper = EarlyStopper(patience=patience, lr_scheduler=self._lr_scheduler)

        loss_fn = loss_fn_param

        # Reduce remaining epochs if restarting from a checkpoint.
        if hasattr(self, 'current_epoch') and self.current_epoch >= 0:

            if num_epochs <= 0:
                print("[WARNING] - No epochs left to train!")
                exit(1)

            print(f"[INFO]: Found already partially trained model."
                  f" - Restarting training from epoch {self.current_epoch + 1}."
                  f" Remaining epochs: {num_epochs - (self.current_epoch + 1)}")

        else:
            print("[INFO]: Starting training from scratch.")
            self.current_epoch = 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            running_loss = 0.0
            count = 0
            len_loader = len(train_loader)

            # Activate graph guidance only after the defined epoch.
            is_graph_active = (epoch + 1) >= EPOCH_GRAPH_INTEGRATION
            # Set the GraphNudger Module to be used.
            self.is_using_nudger = is_graph_active

            print(f"Epoch {epoch + 1}/{num_epochs} (Graph guidance active: {is_graph_active})")

            for images, labels, study_ids in train_loader:
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                study_ids = study_ids.to(self.device)

                # Reset the classifier gradient to None before each batch.
                self.classifier_grad = None

                # Forward pass with graph guidance and nudging enabled if active.
                # This forward pass should update self.base_logits.
                adjusted_logits = self.forward(images, use_graph_guidance=is_graph_active, study_ids=study_ids)

                loss_total, _ = self.compute_total_loss(adjusted_logits, labels, self.signs_found,
                                                        lambda_sim=lambda_sim, lambda_kl=lambda_kl,
                                                        loss_fn_class=loss_fn)
                loss_total.backward()
                optimizer.step()

                # Update of the graph weights is done with ground truth labels
                # This is done only for the first two epochs to avoid useless updates
                if self.current_epoch < 2:
                    for i, study_id in enumerate(study_ids):
                        # Update the graph with the new weights.
                        # This function updates the weights of the edges in the graph based on the classifier logits.
                        self.graph_attention_module.update_edge_weights_with_ground_truth(
                            self.graph, study_id, labels[i].cpu().numpy()
                        )

                running_loss += loss_total.detach().item()

                count += 1
                if self.save_debug_features_log and count % 100 == 0:
                    print(f"[DEBUG] Epoch {epoch + 1}, Batch {count}/{len_loader} - "
                          f"Signs found: {len(self.signs_found)}, "
                          f"Features extracted: {len(self.features_dict_batch)}")

                if count % 1000 == 0:
                    print("Step:", count ," overall steps:", len_loader)

            print(f"[TRAIN] Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / count:.4f}")

            # Save model each epoch to not lose progress.
            self.save_all()

            # Validation step for early stopping verification and lr scheduler step.
            if (use_validation and self._validate_in_training(loss_fn, epoch,
                                            validation_loader=validation_loader)):
                break

        # Set the model back to evaluation mode.
        self.eval()

    def compute_total_loss(self, adjusted_logits, labels, signs_found,
                           loss_fn_class, lambda_sim=0.5, lambda_kl=0.2) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss for graph-augmented Swin training.
        self.classifier_logits: Logits before nudging [B, C]

        Args:
            adjusted_logits: Final logits [B, C]
            labels: Ground truth labels [B, C]
            signs_found: List of dicts with similarity info per sample
            loss_fn_class: BCE or focal loss
            lambda_sim: Weight for attention-based loss
            lambda_kl: Weight for nudging KL loss

        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        bat, _ = labels.shape

        # --- 1. Classification loss (Focal/BCE) ---
        loss_class = loss_fn_class(adjusted_logits, labels)

        # - 2. Attention-focus loss: push the map to have
        # high similarity towards clinical signs
        focus_total = sum((1.0 - torch.tensor(sims, device=self.device).mean()) if sims else 0.0 for sims in
                          ([r["similarity"] for r in signs_found[i]] for i in range(bat)))
        loss_focus = focus_total / bat

        # --- 3. Nudger KL divergence loss ---
        probs_adj = torch.sigmoid(adjusted_logits)
        probs_base = torch.sigmoid(self.classifier_logits)
        probs_adj = torch.clamp(probs_adj, 1e-6, 1.0)
        probs_base = torch.clamp(probs_base, 1e-6, 1.0)

        # KL(input || target), where input is log_probs, target is probs
        loss_kl = fc.kl_div(probs_adj.log(), probs_base, reduction='batchmean') # TODO Check direction of KL divergence

        # --- 4. Combine ---
        total_loss = loss_class + lambda_sim * loss_focus + lambda_kl * loss_kl
        return total_loss, {
            "loss_class": loss_class.item(),
            "loss_attention": loss_focus,
            "loss_kl": loss_kl.item(),
            "total_loss": total_loss.item()
        }

    def _validate_in_training(self, loss_fn, epoch, validation_loader=None):
        # --- Validation step ---
        """
        Validation step for early stopping verification and lr scheduler step.
        Args:
            loss_fn (callable): Loss function to compute the validation loss.
            epoch (int): Current epoch number.
            validation_loader (DataLoader, optional): DataLoader for the validation dataset.
        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if validation_loader is not None:
            self.eval()
            val_running_loss = 0.0
            val_count = 0
            val_labels = []
            val_preds = []

            # Validation loop
            with torch.no_grad():
                for images_val, labels_val in validation_loader:
                    images_val = images_val.to(self.device)
                    labels_val = labels_val.to(self.device)

                    val_logits = self.forward(images_val, use_graph_guidance=True)
                    val_loss = loss_fn(val_logits, labels_val)

                    val_running_loss += val_loss.item()
                    val_count += 1

                    # Compute accuracy and f1 score
                    preds = torch.sigmoid(val_logits).cpu().numpy()

                    val_labels.append(labels_val.cpu().numpy())
                    val_preds.append((preds > 0.5).astype(int))

            # Concatenate all labels and predictions
            val_labels_concat = np.concatenate(val_labels,axis=0)
            val_preds_concat = np.concatenate(val_preds,axis=0)

            val_accuracy = accuracy_score(val_labels_concat, val_preds_concat)
            val_f1 = f1_score(val_labels_concat, val_preds_concat, average='macro')

            print(f"[VAL] Epoch {epoch + 1} - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

            val_loss_epoch = val_running_loss / val_count
            print(f"[VAL] Epoch {epoch + 1} - Validation Loss: {val_loss_epoch:.4f}")
        else:
            val_loss_epoch = None

        # Step the scheduler (even without validation loss, will just do warmup)
        self._lr_scheduler.step(val_loss=val_loss_epoch)

        # Early stopping
        if val_loss_epoch is not None:
            if self._early_stopper.early_stop(val_loss_epoch, self):
                print(f"[INFO] Early Stopping triggered at epoch {epoch + 1}")
                return True

        self.train()  # Set back to training mode
        self.swin_model.train()  # Set the Swin model back to training mode
        return False

    def save_model(self, path=None):
        """
        Save the model to the specified path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')
        torch.save(self, path)

    def load_model_from_state(self, state_dict_path=None, graph_json=None):
        """
        Load the model state dict from the specified path.
        This includes loading the graph and rebuilding the adjacency matrix.
        Args:
            state_dict_path (str, optional): Path to the model file. If None, defaults to 'med_vixray_model_state.pth'.
            graph_json (str, optional): Path to the graph JSON file. If None, defaults to 'med_vixray_model_graph.json'.
        Note:
            - This will override the current model state.
            - The model is set to evaluation mode after loading.
            - The graph is updated with the new weights.
            - The model is set to use nudging module.
        """
        print("Warning: Loading model state dict and graph JSON. This will override the current model state.")

        if state_dict_path is None:
            print(" - State dict path not provided. Loading default state dict.")
            state_dict_path = os.path.join(MODELS_DIR, 'med_vixray_model_state.pth')
            if not os.path.exists(state_dict_path):
                raise FileNotFoundError(f"State dict file not found at {state_dict_path}.")

        if graph_json is None:
            print(" - Graph JSON not provided. Loading default graph JSON.")
            graph_path = os.path.join(MODELS_DIR, 'med_vixray_model_graph.json')
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph JSON file not found at {graph_path}.")
            with open(graph_path, 'r') as graph_file:
                graph_json = json.load(graph_file)

        # Load the state dict from the specified path.
        checkpoint = torch.load(state_dict_path, map_location=self.device, weights_only=False)
        # Load the model state dict.
        self.load_state_dict(checkpoint["model_state_dict"])

        self.graph = graph_json
        print(" - Graph Json Assigned.")

        num_signs = len(self.graph["nodes"]) - len(MIMIC_LABELS)

        if checkpoint.get("graph_matrix") is not None:
            self.graph_matrix = checkpoint["graph_matrix"]
            print(" - Graph Matrix loaded from checkpoint.")
        else:
            print(" - Graph Matrix not found in checkpoint. Building it...")
            self.graph_matrix = build_adjacency_matrix(self.graph,
                                                       num_diseases=len(MIMIC_LABELS),
                                                       num_signs=num_signs)
        # Default: disable training mode.
        # Current Epoch Defaults to 0 as is implied at least 1 epoch was done before saving
        self.current_epoch = checkpoint.get("current_epoch", 0) + 1
        self.is_using_nudger = checkpoint.get("is_using_nudger", True)
        self.is_graph_used = checkpoint.get("is_graph_used", False)
        self.eval()

    def save_state(self, path=None):
        """
        Save the model state.
        This includes the model state dict, graph, and current epoch.

        Args:
            path (str, optional): Path to save the model state. If None, defaults to 'med_vixray_model_state.pth'.
        """
        state = {
            'model_state_dict': self.state_dict(),
            'graph': self.graph,
            'current_epoch': self.current_epoch,
            'graph_matrix': self.graph_matrix,
            'is_graph_used': self.is_graph_used,
            'is_using_nudger': self.is_using_nudger
        }

        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model_state.pth')
        torch.save(state, path)

    def save_graph(self, path=None):
        """
        Save the graph to the specified path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_graph.json')
        with open(path, 'w') as g_file:
            json.dump(self.graph, g_file)

    def save_all(self, path=None):
        """
        Save the model and graph to the specified path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            #self.save_model(path)
        except Exception as e:
            print(f"[ERROR] Error saving whole model: {e}")

        self.save_graph(path.replace('.pth', '_graph.json'))
        print(f"[INFO] Graph saved to {path}")
        self.save_state(path.replace('.pth', '_state.pth'))
        print(f"[INFO] Model state saved to {path.replace('.pth', '_state.pth')}")


if __name__ == "__main__":
    """
        Main function to run the Med-ViX-Ray model training or evaluation.
    """
    print("Starting Med-ViX-Ray model main...")
    print("Loading Graph JSON...")
    # Load the graph JSON file
    with open(MANUAL_GRAPH, 'r') as file:
        data_graph_json = json.load(file)

    # Check for device
    print("Checking for device...")
    t_device = torch.device("cpu" if torch.version.hip else
                            ("cuda" if torch.cuda.is_available() else "cpu"))
    is_cuda = torch.cuda.is_available() and not torch.version.hip
    # ROCm is not supported yet
    print(f"Using device: {t_device}")

    # Init Model
    med_model = SwinMIMICGraphClassifier(graph_json=data_graph_json, device=t_device).to(t_device)
    print("Model initialized.")
    # You can now train the model using the train_model method.
    # Example: model.train_model(train_loader)
    # Note: train_loader should be defined with your training dataset.

    print("Testing attention hook...")
    general.test_attention_hook(med_model.swin_model, t_device)
    print("[INFO] Overwritten attention forward function with hook-enabled version.")

    print(" -- Verifying save directory and loading model state if exists --")
    SAVE_DIR = os.path.join(MODELS_DIR, "med-vix-ray")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # If it still doesn't exist exit, so avoid errors after waiting all training
    if not os.path.exists(SAVE_DIR):
        print("Unable to create save dir. Exiting...")
        exit(1)

    # Load Model if exists
    model_path = os.path.join(SAVE_DIR, "med_vixray_model.pth")
    model_state_path = os.path.join(SAVE_DIR, "med_vixray_model_state.pth")
    json_graph_path = os.path.join(SAVE_DIR, "med_vixray_model_graph.json")

    if os.path.exists(model_state_path):
        print("[INFO] Model State found! Trying to load it...")
        try:
            with open(json_graph_path, 'r') as file:
                data_graph_json = json.load(file)
        except (FileNotFoundError, OSError):
            print("[WARNING] Graph JSON file not found. Using default graph.")
            exit(1)

        med_model.load_model_from_state(state_dict_path=model_state_path, graph_json=data_graph_json)
        print("[INFO] Model State loaded!")

    #if not general.basic_menu_model_option(model_path, med_model):
    #    exit(0)

    # Fetches datasets, labels and create DataLoaders which will handle preprocessing images also.
    training_loader, valid_loader = general.get_dataloaders(return_study_id=True,
                                                            return_val_loader=True,
                                                            pin_memory=is_cuda,
                                                            use_bucket=True, verify_existence=False, full_data=True)

    # Train the model
    print("-- Starting training of Med-ViX-Ray --")
    # NOTE: for other parameters, settings.py defines default values
    med_model.train_model(train_loader=training_loader, validation_loader=valid_loader)

    # Save the model
    print(f"Saving model to {model_path}...")
    med_model.save_model(model_path)
    print(f"Model saved")


    # Evaluate the model
    print(" -- Starting evaluation --")
    # Load the test dataset
    print("Loading test dataset...")
    test_loader = general.get_test_dataloader(full_data=True, pin_memory=is_cuda,
                                              use_bucket=True, verify_existence=False,
                                              channels_mode="RGB")
    print("Test dataset loaded.")

    metrics_dict = med_model.model_evaluation(test_loader, save_stats=False)
    print("Evaluation completed.")
    print("Metrics:", metrics_dict)

    # Save the metrics to a file
    metrics_file = os.path.join(SAVE_DIR, "med-vix_metrics.json")
    with open(metrics_file, 'w') as file:
        file.write(str(metrics_dict))
    print(f"Model Metrics saved to {metrics_file}")

    print("Now saving the model architecture to a file...")
    # Save the model architecture to a file
    model_architecture_file = os.path.join(SAVE_DIR, "model_architecture.txt")
    with open(model_architecture_file, 'w') as f:
        f.write(str(med_model))
    print(f"Model architecture saved to {model_architecture_file}")

    print("Training and evaluation completed.")
    print("Exiting...")
