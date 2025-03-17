import torch
import os
import torch.nn as nn
from settings import NUM_EPOCHS, LEARNING_RATE_TRANSFORMER, LEARNING_RATE_CLASSIFIER, UNBLOCKED_LEVELS, MIMIC_LABELS, MODELS_DIR
from src.fine_tuned_model import SwinMIMICClassifier
from xai.attention_map import AttentionMap
from xai.feature_extract import update_graph_features, extract_heatmap_features

class SwinMIMICGraphClassifier(SwinMIMICClassifier):
    """
    This subclass extends SwinMIMICClassifier by integrating graph-based information.

    It uses an AttentionMap to extract heatmap features during training/inference,
    updates a clinical findings graph based on those features,
    and—after a given number of epochs—incorporates the graph information into the logits.

    The basic idea is to modify the forward pass as:

        base_logits = model(x)
        if training:
            generate attention map, extract features, update graph
            if current_epoch >= threshold:  # after 10-20% of training epochs
                graph_bias = f(graph)  # e.g., computed as: base_logits @ G
                logits = base_logits + graph_bias
            else:
                logits = base_logits
        else:  # inference
            generate attention map, extract features, compute graph bias
            logits = base_logits + graph_bias

    You can later refine how G is computed (e.g., using the formula A′ = softmax(dk QK^T + G)V).
    """

    def __init__(self, num_classes=len(MIMIC_LABELS), graph_json=None,
                 graph_integration_start_epoch=2):
        """
        Args:
            num_classes (int): Number of classes.
            graph_json (dict): JSON structure holding your graph. If None, an empty graph is created.
            graph_integration_start_epoch (int): After which epoch (or fraction of training) to incorporate graph info.
        """
        super(SwinMIMICGraphClassifier, self).__init__(num_classes=num_classes)

        # Epoch threshold after which to add graph contribution (e.g., after 10-20% of training epochs)
        self.graph_integration_start_epoch = graph_integration_start_epoch
        self.current_epoch = 0  # To be updated during training

        # Initialize the graph; if none is provided, start with an empty graph
        if graph_json is None:
            # A simple placeholder: you may want to initialize nodes and edges as needed.
            self.graph = {"nodes": [], "edges": []}
        else:
            self.graph = graph_json

        # Initialize the AttentionMap instance using the current backbone.
        # (We use gradcam here—but you can choose "cdam" if/when it’s fully implemented.)
        self.attention_map_generator = AttentionMap(model=self.swin_model, xai_type="gradcam")

    def compute_graph_bias(self, base_logits):
        """
        Compute a graph-based bias from the graph structure.
        For example, if your graph contains an adjacency matrix, you can compute:

            bias = alpha * (base_logits @ G)

        where G is the (num_classes x num_classes) matrix. If G is not directly available,
        we compute it from the edges (assuming symmetric connections).

        Args:
            base_logits (torch.Tensor): Tensor of shape (batch, num_classes)
        Returns:
            torch.Tensor: A bias tensor that can be added to base_logits.
        """
        # Try to use an existing adjacency matrix if available.
        if "adjacency" in self.graph:
            g_val = torch.tensor(self.graph["adjacency"], dtype=torch.float32, device=self.device)
        else:
            # Build a symmetric adjacency matrix from the graph's edges.
            num_nodes = len(self.graph["nodes"])
            g_val = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)
            for edge in self.graph.get("edges", []):
                i = int(edge["source"])
                j = int(edge["target"])
                weight = edge.get("weight", 1.0)
                g_val[i, j] = weight
                g_val[j, i] = weight  # assume undirected
        # Compute a simple bias: base_logits @ G
        graph_bias = base_logits @ g_val
        # Scale the bias (alpha is a hyperparameter to be tuned)
        alpha = 0.1
        return alpha * graph_bias

    def forward(self, x):
        """
        In forward(), we first compute the base logits using the parent class.
        Then, if in training mode, we generate the attention map, extract heatmap features,
        and update the graph for the corresponding clinical finding(s).
        If the current epoch is beyond the integration threshold, we adjust the logits
        using the graph bias.
        In inference mode, we always add the graph-based adjustment.
        """
        base_logits = super(SwinMIMICGraphClassifier, self).forward(x)

        # Always generate the attention map and extract features.
        att_map = self.attention_map_generator.generate_attention_map(self.swin_model, x)
        features = extract_heatmap_features(att_map)

        # Update the graph with these features.
        # Here, we assume a given clinical finding node label; adjust as needed.
        self.graph = update_graph_features(self.graph, extracted_features=features, sign_label="example_sign")

        # If training and before the integration threshold, use the base logits only.
        if self.training:
            if self.current_epoch < self.graph_integration_start_epoch:
                return base_logits
            else:
                # After the threshold epoch, integrate graph information.
                graph_bias = self.compute_graph_bias(base_logits)
                adjusted_logits = base_logits + graph_bias
                return adjusted_logits
        else:
            # In inference, integrate graph information.
            graph_bias = self.compute_graph_bias(base_logits)
            adjusted_logits = base_logits + graph_bias
            return adjusted_logits

    def train_model(self, train_loader, num_epochs=NUM_EPOCHS,
                    learning_rate_swin=LEARNING_RATE_TRANSFORMER,
                    learning_rate_classifier=LEARNING_RATE_CLASSIFIER,
                    layers_to_unblock=UNBLOCKED_LEVELS, optimizer_param=None,
                    loss_fn_param=nn.BCEWithLogitsLoss()):
        """
        Override train_model to update the current epoch counter.
        This method calls the parent training loop for one epoch at a time.
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch  # update epoch counter
            print(
                f"Epoch {epoch + 1}/{num_epochs} (Graph integration starts at epoch {self.graph_integration_start_epoch})")
            super(SwinMIMICGraphClassifier, self).train_model(
                train_loader,
                num_epochs=1,
                learning_rate_swin=learning_rate_swin,
                learning_rate_classifier=learning_rate_classifier,
                layers_to_unblock=layers_to_unblock,
                optimizer_param=optimizer_param,
                loss_fn_param=loss_fn_param
            )

    def save_model(self, path=None):
        """
        Save the model to the specified path.
        If no path is provided, it saves the model to the default path in the models directory with the name 'med_vixray_model.pth'.
        :param path: Path to save the model: It expects a string with the path and filename (e.g., 'model.pth').
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')

        super().save_model(path)

    def load_model(self, path=None):
        """
        Load the model from the specified path.
        If no path is provided, it loads the model from the default path in the models directory with the name 'med_vixray_model.pth'.
        :param path: Path to load the model: It expects a string with the path and filename (e.g., 'model.pth').
        """
        if path is None:
            path = os.path.join(MODELS_DIR, 'med_vixray_model.pth')

        super().load_model(path)
