import json
import torch
import os
import torch.nn as nn
import numpy as np

from settings import NUM_EPOCHS, LEARNING_RATE_TRANSFORMER, LEARNING_RATE_CLASSIFIER, UNBLOCKED_LEVELS, MIMIC_LABELS, \
    MODELS_DIR
from src import general
from src.fine_tuned_model import SwinMIMICClassifier
from xai.attention_map import AttentionMap
from xai.feature_extract import update_graph_features, extract_heatmap_features


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
        # Add the graph matrix (broadcasted along batch dimension)
        score = score + self.G.unsqueeze(0)
        # Apply softmax to obtain attention weights.
        atten = torch.softmax(score, dim=-1)
        # Aggregate the embeddings according to the attention weights.
        aggr = torch.bmm(atten, e_img)
        # Project each aggregated embedding to a scalar bias.
        bias = self.W_out(aggr).squeeze(-1)
        return bias


# ========================= SWIN MIMIC + GRAPH CLASSIFIER =========================


class SwinMIMICGraphClassifier(SwinMIMICClassifier):
    """
    This subclass extends SwinMIMICClassifier by integrating graph-based information.
    It uses an AttentionMap to extract heatmap features during training/inference,
    updates a clinical findings graph based on those features,
    and—after a given number of epochs—incorporates the graph information into the logits.
    """
    def __init__(self, num_classes=len(MIMIC_LABELS), graph_json=None,
                 graph_integration_start_epoch=2, d_k=64):
        """
        Args:
            num_classes (int): Number of classes.
            graph_json (dict): JSON structure holding your graph. If None, an empty graph is created.
            d_k (int): Dimension for the key/query/value embeddings.
            graph_integration_start_epoch (int): After which epoch to incorporate graph info.
        """
        super(SwinMIMICGraphClassifier, self).__init__(num_classes=num_classes)

        self.graph_integration_start_epoch = graph_integration_start_epoch
        self.current_epoch = 0

        # Initialize the graph.
        if graph_json is None:
            self.graph = {"nodes": [], "edges": []}
            graph_matrix = None
        else:
            self.graph = graph_json
            num_signs = len(graph_json["nodes"]) - len(MIMIC_LABELS)
            graph_matrix = build_adjacency_matrix(
                graph_json, num_diseases=len(MIMIC_LABELS), num_signs=num_signs
            )

        # Initialize the AttentionMap.
        self.attention_map_generator = AttentionMap(model=self.swin_model, xai_type="cdam")
        # Initialize the GraphAttentionModule.
        self.graph_attention_module = GraphAttentionModule(num_classes=num_classes, d_k=d_k, graph_matrix=graph_matrix)

    def compute_graph_bias(self, base_logits):
        """
        Compute a refined graph bias using the GraphAttentionModule.
        """
        return self.graph_attention_module(base_logits)

    def forward(self, x):
        """
        Compute base logits using the parent class, then update the graph
        and, if applicable, adjust the logits with the graph bias.
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
