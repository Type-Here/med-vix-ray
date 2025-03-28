import json
import torch
import os
import torch.nn as nn
import numpy as np

from settings import NUM_EPOCHS, LEARNING_RATE_TRANSFORMER, LEARNING_RATE_CLASSIFIER, UNBLOCKED_LEVELS, MIMIC_LABELS, \
    MODELS_DIR, LAMBDA_REG, EPOCH_GRAPH_INTEGRATION, ALPHA_GRAPH, ATTENTION_MAP_THRESHOLD
from src import general
from src.fine_tuned_model import SwinMIMICClassifier

import xai.attention_map as attention
import xai.feature_extract as xai_fe


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
    if keys_order is not None:
        # Gather the feature values; if a key is missing, use 0.0 as default.
        feature_list = [features_dict.get(k, 0.0) for k in keys_order]
    else:
        # If no keys are provided, use the order of the features in the dictionary.
        feature_list = [features_dict.get(k, 0.0) for k in features_dict.keys()]

    for key in keys_order:
        # Ensure each feature tensor has shape [B, 1]
        tensor_val = features_dict[key].view(-1, 1)
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
    # Initialize the blocks with float32 type.
    matr_dd = np.zeros((num_diseases, num_diseases), dtype=np.float32)  # disease-disease
    matr_ds = np.zeros((num_diseases, num_signs), dtype=np.float32)  # disease-sign

    # Process each edge in the JSON
    for edge in graph_json["edges"]:
        weight = edge.get("weight", 1.0)
        if edge["type"] == "correlation":
            i = int(edge["source"])  # disease index
            j = int(edge["target"])  # disease index
            # Scale correlation edges
            matr_dd[i, j] = weight * scale_corr
            matr_dd[j, i] = weight * scale_corr  # if undirected
        elif edge["type"] == "finding":
            # Assume the source is a disease and target is a sign node.
            i = int(edge["source"])  # disease index
            k = int(edge["target"])  # sign index
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

# ========================= GRAPH ATTENTION BIAS =========================


class GraphAttentionBias(nn.Module):
    def __init__(self, alpha=ALPHA_GRAPH):
        super(GraphAttentionBias, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))  # or fix as a hyperparameter

    def forward(self, attn_scores, graph_adj_matrix):
        # attn_scores: [B, H, N, N], graph_adj_matrix: [N, N]
        # Expand graph matrix to batch and head dims:
        g_expanded = graph_adj_matrix.unsqueeze(0).unsqueeze(0)  # shape [1,1,N,N]
        modified_scores = attn_scores / (attn_scores.shape[-1]**0.5) + self.alpha * g_expanded
        return torch.softmax(modified_scores, dim=-1)


# ========================= GRAPH ATTENTION MODULE =========================


class GraphAttentionModule(nn.Module):
    def __init__(self, num_classes, d_k, graph_matrix=None):
        """
        Initialize the graph attention module.

        Args:
            num_classes (int): Number of classes (or graph nodes).
            d_k (int): Dimension for the key/query/value embeddings.
            graph_matrix (np.array or torch.Tensor, optional): The pre-computed graph matrix (shape: num_classes x num_classes).
                If None, defaults to a zero matrix.
        """
        super(GraphAttentionModule, self).__init__()
        self.num_classes = num_classes
        self.d_k = d_k

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


# ========================= GRAPH NUDGER =========================


class GraphNudger(nn.Module):
    def __init__(self, eta=0.01):
        super(GraphNudger, self).__init__()
        self.eta = eta  # Nudging learning rate

    def forward(self, heatmap_features_batch, keys_order, graph, num_diseases, grad_output_batch):
        """
        Compute a nudging bias vector for each sample in the batch based on the difference
        between the extracted heatmap features and the stored features in the graph's sign nodes.

        For each sample and for each disease node d, we sum over all finding edges from d to sign nodes:

            Δb_d^(i) = η * ∑_{edge: source=d, type='finding'} (w_{d,s} * sim(f_att^(i), f_s) * g^(i))

        where:
          - f_att^(i) is the heatmap feature vector for sample i,
          - f_s is the stored feature vector for the sign node,
          - sim(·,·) is the cosine similarity (normalized to [0,1]),
          - g^(i) is the gradient vector for sample i (elementwise used),
          - w_{d,s} is the edge weight.

        Args:
            heatmap_features_batch (torch.Tensor): [B, f_dim] tensor of extracted features from the attention map.
            keys_order (list): List of keys specifying the order in which features should appear.
            graph (dict): The entire graph with keys "nodes" and "edges". Each sign node in graph["nodes"]
                          is expected to have "id", "type"=="sign", and "features" (a dict).
            num_diseases (int): Number of disease (label) nodes.
            grad_output_batch (torch.Tensor): [B, f_dim] tensor of gradients from the classifier head.

        Returns:
            torch.Tensor: A tensor of shape [B, num_diseases] containing the nudging bias for each sample.
        """
        nudge_batch_size, f_dim = heatmap_features_batch.shape
        # Initialize an empty update tensor for the batch.
        update_list = []

        # Process each sample in the batch.
        for i in range(nudge_batch_size):
            # Get the current sample's heatmap feature vector (as a NumPy array).
            current_vec = heatmap_features_batch[i].cpu().numpy()
            # Get the corresponding gradient vector (as numpy array).
            grad_value = grad_output_batch[i].cpu().numpy()  # shape: [f_dim]
            # Initialize a bias vector for diseases (shape: [num_diseases]).
            bias_vec = np.zeros(num_diseases, dtype=np.float32)
            # Loop over each edge in the graph of type 'finding'.
            for edge in graph["edges"]:
                if edge["type"] == "finding":
                    disease_idx = int(edge["source"])  # disease node index
                    sign_idx = int(edge["target"])  # sign node index
                    weight = edge.get("weight", 1.0)
                    self.__compute_bias_dinamically(bias_vec, current_vec, disease_idx,
                                                   graph, keys_order, sign_idx,
                                                   weight, grad_value)
            # Convert bias vector to tensor and append.
            update_list.append(torch.tensor(bias_vec, dtype=torch.float32, device=heatmap_features_batch.device))

        # Stack updates to form a tensor of shape [B, num_diseases].
        update_tensor = torch.stack(update_list, dim=0)
        return update_tensor

    def __compute_bias_dinamically(self, bias_vec, current_vec, disease_idx,
                                   graph, keys_order, sign_idx, weight, grad_value=None):
        """
        Compute the bias for a specific disease node based on the cosine similarity.
        The bias_vec is updated in place.
        Args:
            bias_vec (np.ndarray): The bias vector to be updated.
            current_vec (np.ndarray): The current heatmap feature vector.
            disease_idx (int): The index of the disease node.
            graph (dict): The entire graph with keys "nodes" and "edges".
            keys_order (list): List of keys specifying the order in which features should appear.
            sign_idx (int): The index of the sign node.
            weight (float): The weight of the edge between the disease and sign node.
        """
        # Find the corresponding sign node.
        for node in graph["nodes"]:
            if node.get("type") == "sign" and int(node["id"]) == sign_idx:
                stored_features = node.get("features", None)
                if stored_features is None:
                    return  # No features available for this sign node.

                # Convert stored features dict to vector.
                stored_vec = _compute_feature_vector(stored_features, keys=keys_order)  # numpy array
                # Compute cosine similarity.
                sim = xai_fe.__cosine_similarity(stored_vec, current_vec)
                # Normalize similarity to [0,1].
                sim = (sim + 1.0) / 2.0

                # Incorporate gradient information.
                # Use the gradient info from the classifier head backpropagation.
                grad_factor = np.linalg.norm(grad_value)

                # Accumulate contribution for this disease.
                bias_vec[disease_idx] += self.eta * weight * sim * grad_factor
            break  # sign node found, break inner loop


# ========================= SWIN MIMIC + GRAPH CLASSIFIER =========================


class SwinMIMICGraphClassifier(SwinMIMICClassifier):
    """
    This subclass extends SwinMIMICClassifier by integrating graph-based information.
    It uses an AttentionMap to extract heatmap features during training/inference,
    updates a clinical findings graph based on those features,
    and—after a given number of epochs—incorporates the graph information into the logits.
    """
    def __init__(self, num_classes=len(MIMIC_LABELS), graph_json=None,
                 graph_integration_start_epoch=EPOCH_GRAPH_INTEGRATION, d_k=64):
        """
        Args:
            num_classes (int): Number of classes.
            graph_json (dict): JSON structure holding your graph. If None, an empty graph is created.
            d_k (int): Dimension for the key/query/value embeddings.
            graph_integration_start_epoch (int): After which epoch to incorporate graph info.
        """
        super(SwinMIMICGraphClassifier, self).__init__(num_classes=num_classes)

        # Placeholder for base logits, to be used for graph bias computation.
        self.base_logits = None
        # Placeholder for the classifier logits, to be used for loss computation.
        self.classifier_logits = None
        # Placeholder for the classifier gradient, to be used for nudging backpropagation.
        self.classifier_grad = None
        # Placeholder for graph features list to be used for nudging. Will be filled on first call needed.
        self.stats_keys = None

        # Flag indicating whether graph guidance has been injected into transformer layers.
        self.is_graph_used = False

        # Epoch threshold to activate graph-based mechanisms (both transformer bias & final nudging).
        self.graph_integration_start_epoch = graph_integration_start_epoch
        # Current epoch counter (to be updated during training).
        self.current_epoch = 0
        # Total epochs; useful for calculating activation timing.
        self.total_epochs = NUM_EPOCHS

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

        # Initialize the AttentionMap module.
        # This module will extract attention maps from the second last layer (before the classifier head)
        # using techniques such as cdam (or gradcam) to later compare with sign node statistics.
        self.attention_map_generator = attention.AttentionMap(model=self.swin_model, xai_type="cdam")

        # Initialize the GraphAttentionModule.
        # This module uses the (normalized) adjacency matrix (from both correlation and finding edges)
        # to compute a bias that will be injected into transformer layers.
        self.graph_attention_module = GraphAttentionModule(num_classes=num_classes, d_k=d_k, graph_matrix=self.graph_matrix)

        # Initialize the GraphNudger module.
        # This module is responsible for the final nudging operation on the classifier head:
        # it compares attention-derived features with stored sign node statistics and computes a weight update.
        self.graph_nudger = GraphNudger(eta=0.01)  # nudging learning rate

        # Note: self.classifier is already defined in the parent class (SwinMIMICClassifier).

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
        if not use_graph_guidance:
            return  # do nothing

        # Ensure graph_adj_matrix is a torch tensor:
        if not torch.is_tensor(graph_adj_matrix):
            graph_adj_matrix = torch.tensor(graph_adj_matrix, dtype=torch.float32,
                                            device=self.swin_model.patch_embed.proj.weight.device)

        graph_bias_module = GraphAttentionBias(alpha=ALPHA_GRAPH)
        # Assume self.swin_model.layers is a list of layers, each with blocks that have an "attn" module.
        for layer_idx, layer in enumerate(self.swin_model.layers):
            for block_idx, block in enumerate(layer.blocks):
                original_forward = block.attn.forward

                def new_forward(x, *args, orig_forward=original_forward, **kwargs):
                    # Get raw attention scores (this requires that the original attn returns them)
                    attn_scores = orig_forward(x, *args, **kwargs)
                    # Inject graph bias using our module. Expecting attn_scores to be of shape [B, H, N, N]

                    modified_attn = graph_bias_module(attn_scores, graph_adj_matrix)
                    return modified_attn

                block.attn.forward = new_forward
                print(f"Injected graph bias into layer {layer_idx}, block {block_idx}")

    def forward(self, x, use_graph_guidance=True, use_nudger=False):
        """
        base_logits = super(SwinMIMICGraphClassifier, self).forward(x)
        att_map = self.attention_map_generator.generate_attention_map(self.swin_model, x)
        features = extract_heatmap_features(att_map)
        self.graph = update_graph_features(self.graph, extracted_features=features, sign_label="example_sign")

        if self.training:
            if self.current_epoch < self.graph_integration_start_epoch:
                return base_logits
            else:
                graph_bias = self.compute_graph_bias(base_logits)
                return base_logits + graph_bias
        else:
            graph_bias = self.compute_graph_bias(base_logits)
            return base_logits + graph_bias

    def train_model(self, train_loader, num_epochs=NUM_EPOCHS,
                    learning_rate_swin=LEARNING_RATE_TRANSFORMER,
                    learning_rate_classifier=LEARNING_RATE_CLASSIFIER,
                    layers_to_unblock=UNBLOCKED_LEVELS, optimizer_param=None,
                    loss_fn_param=nn.BCEWithLogitsLoss(), lambda_reg=0.1):
        """
        Custom training loop that incorporates the graph-based loss regularization.
        """
        self.__unblock_layers(layers_to_unblock)

        # Define optimizer with parameter groups.
        optimizer = self.__create_optimizer(layers_to_unblock, learning_rate_swin,
                                            learning_rate_classifier, optimizer_param)
        loss_fn = loss_fn_param

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            running_loss = 0.0
            count = 0

            print(f"Epoch {epoch + 1}/{num_epochs} (Graph integration starts at epoch {self.graph_integration_start_epoch})")
            for images, labels in train_loader:
                optimizer.zero_grad()
                # Compute base logits (without graph bias) using the parent forward.
                base_logits = super(SwinMIMICGraphClassifier, self).forward(images)
                # Compute adjusted logits using our forward (which incorporates graph bias if applicable).
                adjusted_logits = self.forward(images)
                # Determine graph_bias if graph integration is active.
                if self.current_epoch >= self.graph_integration_start_epoch:
                    graph_bias = adjusted_logits - base_logits
                else:
                    graph_bias = torch.zeros_like(base_logits)

                # Compute the classification loss.
                loss_class = loss_fn(adjusted_logits, labels)
                # Compute the regularization loss (L2 norm on the graph bias).
                loss_reg = lambda_reg * torch.mean(graph_bias ** 2)
                loss_total = loss_class + loss_reg

                loss_total.backward()
                optimizer.step()

                running_loss += loss_total.item()
                count += 1

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / count:.4f}")
            # Save model each epoch to not lose progress.
            self.save_all()

    def save_model(self, path=None):
        """
        Save the model to the specified path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')
        super().save_model(path)

    def load_model(self, path=None):
        """
        Load the model from the specified path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')
        super().load_model(path)

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
            'current_epoch': self.current_epoch
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
        with open(path, 'w') as file:
            json.dump(self.graph, file)

    def save_all(self, path=None):
        """
        Save the model and graph to the specified path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')
        self.save_model(path)
        self.save_graph(path.replace('.pth', '_graph.json'))
        self.save_state(path.replace('.pth', '_state.pth'))


if __name__ == "__main__":
    """
        Main function to run the Med-ViX-Ray model training or evaluation.
    """
    print("Starting Med-ViX-Ray model main...")
    # Example usage
    med_model = SwinMIMICGraphClassifier()
    print("Model initialized.")
    # You can now train the model using the train_model method.
    # Example: model.train_model(train_loader)
    # Note: train_loader should be defined with your training dataset.

    SAVE_DIR = os.path.join(MODELS_DIR, "med-vix-ray")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load Model if exists
    model_path = os.path.join(SAVE_DIR, "med_model.pth")

    if not general.model_option(model_path, med_model):
        exit(0)

    # Fetches datasets, labels and create DataLoaders which will handle preprocessing images also.
    training_loader, valid_loader = general.get_dataloaders()

    # Train the model
    print("Starting training of Med-ViX-Ray...")
    # NOTE: for other parameters, settings.py defines default values
    med_model.train_model(training_loader)

    # Save the model
    print(f"Saving model to {model_path}...")
    med_model.save_model(model_path)
    print(f"Model saved")

    # Evaluate the model
    print("Starting evaluation...")
    metrics_dict = med_model.model_evaluation(valid_loader, save_stats=False)
    print("Evaluation completed.")
    print("Metrics:", metrics_dict)

    # Save the metrics to a file
    metrics_file = os.path.join(SAVE_DIR, "fine_tuned_metrics.json")
    with open(metrics_file, 'w') as f:
        f.write(str(metrics_dict))
    print(f"Metrics saved to {metrics_file}")

    # Save the model architecture to a file
    model_architecture_file = os.path.join(SAVE_DIR, "model_architecture.txt")
    with open(model_architecture_file, 'w') as f:
        f.write(str(med_model))
    print(f"Model architecture saved to {model_architecture_file}")

    print("Training and evaluation completed.")
    print("Exiting...")
