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
            model: Swin V2 src.
            xai_type: Type of attention map to generate (default is "cdam").
        """
        if model is None:
            raise ValueError("Model must be provided.")

        self.map = None
        self.mask = None

        self.type = xai_type
        if xai_type not in ["cdam", "gradcam"]:
            raise ValueError("Invalid type. Choose 'cdam' or 'gradcam'.")
        if xai_type == "cdam":
            print("⚠️ CDAM is not fully implemented yet. Using Grad-CAM instead.")
            self.__set_cdam(model)
        else:
            print("✅ Using Grad-CAM for attention map generation.")
            self.__set_grad_cam(model)

    # -------- SETTER: CHANE XAI TYPE --------
    def set_generation_method(self, method, model):
        """
        Setter: Change the method for generating attention/heat-maps.
        Args:
            method: Method to use for generating maps. Choose "cdam" or "gradcam".
            model: Swin V2 src.
        Raises:
            ValueError: If the method is not recognized.
        """
        if method == "cdam":
            self.__set_cdam(model)
        elif method == "gradcam":
            self.__set_grad_cam(model)
        else:
            raise ValueError("Invalid method. Choose 'cdam' or 'gradcam'.")

    # -------- INTERNAL SETTERS --------

    def __set_grad_cam(self, model):
        """
            Set the Grad-CAM src.
        """
        # Get the target layer for Grad-CAM
        target_layer = model.swin.layers[-1].blocks[-1].attn

        #self.grad_cam = LayerGradCam(src, src.swin.layers[-1].blocks[-1].attn.proj)
        # Create a LayerGradCam object
        self.grad_cam = LayerGradCam(model, target_layer.proj)

        self.generate_attention_map = self.__generate_attention_map_grad_cam

    def __set_cdam(self, model):
        """
            Set the CDAM src.
        """
        self.cdam = LayerAttribution(model, model.swin.layers[-1].blocks[-1].attn.proj)
        self.generate_attention_map = self.__extract_attention_cdam

    # ============== GRAD-CAM =================

    def __generate_attention_map_grad_cam(self, model, image_tensor):
        """
        Generate an attention map using Grad-CAM.
        Args:
            model: Swin V2 src.
            image_tensor: Pre-processed image tensor (1, 1, 256, 256).
        Returns:
            Heatmap of attention areas.
        """
        # Ensure the src is in evaluation mode
        model.eval()

        # Calculate the Grad-CAM attribution
        attribution = self.grad_cam.attribute(image_tensor, target=0)

        # Convert to numpy and normalize (heatmap)
        heatmap = attribution.squeeze().cpu().detach().numpy()

        # Cast to float32 for compatibility with OpenCV
        # Uncomment if needed
        # heatmap = heatmap.astype(np.float32)

        heatmap_norm = np.zeros_like(heatmap)  # Create an empty array for normalization with same shape
        res_heatmap = cv2.normalize(heatmap_norm, heatmap, 0.0, 255.0, cv2.NORM_MINMAX) # Normalize in place
        self.map = res_heatmap

        return heatmap_norm


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

    def __extract_attention_cdam(self, model, image_tensor, layer_num=-1):
        """
        Extracts attention weights from a specific layer of Swin V2.

        Args:
            model: Pre-trained Swin V2 src.
            image_tensor: Pre-processed image tensor (1, 1, 256, 256).
            layer_num: Transformer layer index (-1 for last layer).

        Returns:
            Attention heatmap.
        """
        model.eval()

        # Get the attention weights from the last transformer layer
        # transformer_layer = src.swin.layers[layer_num].blocks[-1].attn.attn_probs
        transformer_layer = model.swin.layers[layer_num].blocks[-1].attn.get_attn()

        # Forward pass to get the attention weights
        with torch.no_grad():
            # Here we need to pass the image tensor through the src to get the attention weights
            _ = model(image_tensor)

        attention_map = transformer_layer.cpu().detach().numpy()

        # Adds up all attention heads (mean)
        attention_map = np.mean(attention_map, axis=0)

        # Resize to 256x256 (adapt to input size)
        attention_map = cv2.resize(attention_map, (256, 256))

        # Cast to float32 for compatibility with OpenCV
        # Uncomment if needed
        # heatmap = heatmap.astype(np.float32)

        # Normalize the attention map to [0, 255]
        attmap_norm = np.zeros_like(attention_map)  # Create an empty array for normalization with same shape
        res_map = cv2.normalize(attention_map, attmap_norm, 0, 255, cv2.NORM_MINMAX) # Normalize in place
        self.map = res_map

        return attention_map

    # ========= ANALYSIS =============

    def analyze_intensity(self, image, is_3d=False):
        """
        Analyze pixel intensity within the attention map region.
        Args:
            image: Grayscale X-ray image.
            is_3d: If True, squash the heatmap to 2D using mean. Default is False.
        Returns:
            Mean and variance of pixel intensity.
        Raises:
            ValueError: If no attention map is generated yet.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        if is_3d:
            heatmap = np.mean(self.map, axis=0) / np.max(np.mean(self.map, axis=0))
        else:
            # Normalize the heatmap to [0, 1]
            heatmap = self.map / np.max(self.map)

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


    # ======== VISUALIZATION ============

    def visualize_attention(self, image_tensor):
        """
        Visualize the attention map on the original image.
        Args:
            image_tensor: Pre-processed image tensor (1, 1, 256, 256).
        Raises:
            ValueError: If no attention map is generated yet.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        # Convert the image tensor to numpy
        image = image_tensor.squeeze().cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # Change to HWC format

        # Resize the attention map to match the original image size
        attention_map = cv2.resize(self.map, (image.shape[1], image.shape[0]))

        # Normalize the attention map to [0,1] if the image is also in this range
        if np.max(image) <= 1.0:
            attention_map = attention_map / 255.0
        # Normalize the attention map
        # attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)

        # Overlay the attention map on the original image
        overlay = cv2.addWeighted(image.astype(np.uint8), 0.5, attention_map.astype(np.uint8), 0.5, 0)

        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    def visualize_attention_map_only(self):
        """
        Visualize the attention map only. Prefer self.visualize_attention() for better results.
        Raises:
            ValueError: If no attention map is generated yet.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        plt.imshow(self.map, cmap='jet')
        plt.colorbar()
        plt.show()

    # ========== ADJUSTMENT =============

    def adjust_attention_map(self, mean_intensity, pathology):
        """
        Adjust the attention map if it doesn't match expected clinical HU values.
        Args:
            mean_intensity: Mean intensity of the attention map region.
            pathology: Pathology type.
        Raises:
            ValueError: If no attention map is generated yet.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        # TODO: Implement a more sophisticated adjustment
        if self.compare_with_clinical_values(mean_intensity, np.var(self.map),
                                             pathology) == "❌ Potential inconsistency":
            # Reduce the attention map values only in the region of interest
            self.map[self.map > ATTENTION_MAP_THRESHOLD] *= 0.7
        return self.map
