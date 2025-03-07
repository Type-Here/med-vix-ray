import torch
import cv2 # For image processing, resize, and normalization
import numpy as np
import matplotlib.pyplot as plt

from captum.attr import LayerGradCam, LayerAttribution
from settings import ATTENTION_MAP_THRESHOLD


class AttentionMap:
    def __init__(self, model, xai_type="cdam"):
        """
        Initialize the AttentionMap class.
        Args:
            model: Swin V2 model.
            xai_type: Type of attention map to generate (default is "cdam").
        """
        if model is None:
            raise ValueError("Model must be provided.")

        self.type = xai_type
        if xai_type not in ["cdam", "gradcam"]:
            raise ValueError("Invalid type. Choose 'cdam' or 'gradcam'.")
        if xai_type == "cdam":
            print("⚠️ CDAM is not fully implemented yet. Using Grad-CAM instead.")
            self.__set_cdam(model)
        else:
            print("✅ Using Grad-CAM for attention map generation.")
            self.__set_grad_cam(model)

    # ============== GRAD-CAM =================

    def __set_grad_cam(self, model):
        """
        Set the Grad-CAM model.
        """
        self.grad_cam = LayerGradCam(model, model.swin.layers[-1].blocks[-1].attn.proj)
        self.generate_attention_map = self.__generate_attention_map_grad_cam
        self.analyze_intensity = self.__analyze_intensity_grad_cam

    def __generate_attention_map_grad_cam(self, model, image_tensor):
        """
        Generate an attention map using Grad-CAM.
        Args:
            model: Swin V2 model.
            image_tensor: Pre-processed image tensor (1, 1, 256, 256).
        Returns:
            Heatmap of attention areas.
        """
        model.eval()

        # Calculate the Grad-CAM attribution
        attribution = self.grad_cam.attribute(image_tensor, target=0)  # target=0 for the first class

        # Convert to numpy and normalize (heatmap)
        heatmap = attribution.squeeze().cpu().detach().numpy()
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return heatmap

    def __analyze_intensity_grad_cam(self, image, heatmap):
        """
        Analyze pixel intensity within the attention map region.
        Args:
            image: Grayscale X-ray image.
            heatmap: Attention heatmap.
        Returns:
            Mean and variance of pixel intensity.
        """
        # Normalize the heatmap to [0, 1]
        heatmap = heatmap / np.max(heatmap)

        # Create a mask by selecting pixels with high attention
        self.mask = heatmap > ATTENTION_MAP_THRESHOLD  # Threshold

        # Extract the intensity values from the original image
        intensities = image[self.mask]

        # Calculate mean and variance of the intensities
        mean_intensity = np.mean(intensities)
        variance_intensity = np.var(intensities)

        # Inference of approximal shape
        # TODO: Implement a more sophisticated shape inference

        return mean_intensity, variance_intensity

    def compare_with_clinical_values(self, mean_intensity, variance_intensity, pathology):
        """
        Compare intensity values with known clinical ranges.
        """
        # TODO: Implement a more sophisticated comparison with clinical data
        clinical_ranges = {
            "consolidation": {"mean_range": (-100, 100), "variance_range": (0, 50)},
            "ground_glass_opacity": {"mean_range": (-700, -500), "variance_range": (50, 150)},
            "normal": {"mean_range": (-800, -700), "variance_range": (0, 30)},
        }

        if pathology in clinical_ranges:
            mean_range = clinical_ranges[pathology]["mean_range"]
            variance_range = clinical_ranges[pathology]["variance_range"]

            mean_match = mean_range[0] <= mean_intensity <= mean_range[1]
            variance_match = variance_range[0] <= variance_intensity <= variance_range[1]

            if mean_match and variance_match:
                return "✅ Consistency with clinical data"
            else:
                return "❌ Potential inconsistency - Model attention may be incorrect"
        else:
            return "⚠️ No clinical data available for this pathology"

    # ========== CDAM =============
    def __set_cdam(self, model):
        """
        Set the CDAM model.
        """
        self.cdam = LayerAttribution(model, model.swin.layers[-1].blocks[-1].attn.proj)
        self.generate_attention_map = self.__extract_attention_cdam
        self.analyze_intensity = self.__analyze_intensity_grad_cam

    def __extract_attention_cdam(self, model, image_tensor, layer_num=-1):
        """
        Extracts attention weights from a specific layer of Swin V2.

        Args:
            model: Pre-trained Swin V2 model.
            image_tensor: Pre-processed image tensor (1, 1, 256, 256).
            layer_num: Transformer layer index (-1 for last layer).

        Returns:
            Attention heatmap.
        """
        model.eval()

        # Prendiamo l'ultimo blocco di Swin
        transformer_layer = model.swin.layers[layer_num].blocks[-1].attn.attn_probs

        # Forward pass per ottenere l'attenzione
        with torch.no_grad():
            _ = model(image_tensor)  # Inneschiamo il passaggio in avanti per ottenere le attenzioni

        attention_map = transformer_layer.cpu().detach().numpy()

        # Sommiamo su tutte le teste di attenzione
        attention_map = np.mean(attention_map, axis=0)

        # Ridimensioniamo la heatmap alla dimensione dell'immagine
        attention_map = cv2.resize(attention_map, (256, 256))

        # Normalizziamo la heatmap tra 0 e 255
        attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)

        return attention_map


    # ======== VISUALIZATION ============
    def visualize_attention(self, image_tensor, attention_map):
        """
        Visualize the attention map on the original image.
        Args:
            image_tensor: Pre-processed image tensor (1, 1, 256, 256).
            attention_map: Attention heatmap.
        """
        # Convert the image tensor to numpy
        image = image_tensor.squeeze().cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # Change to HWC format

        # Resize the attention map to match the original image size
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Normalize the attention map
        attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)

        # Overlay the attention map on the original image
        overlay = cv2.addWeighted(image.astype(np.uint8), 0.5, attention_map.astype(np.uint8), 0.5, 0)

        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    def visualize_attention_map_only(self, attention_map):
        plt.imshow(attention_map, cmap='jet')
        plt.colorbar()
        plt.show()

    # ========== ADJUSTMENT =============

    def adjust_attention_map(self, heatmap, mean_intensity, pathology):
        """
        Adjust the attention map if it doesn't match expected clinical HU values.
        """
        if self.compare_with_clinical_values(mean_intensity, np.var(heatmap), pathology) == "❌ Potential inconsistency":
            heatmap = heatmap * 0.7  # Reduce the intensity of the attention map
        return heatmap
