import torch
import cv2  # For image processing, resize, and normalization
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

    # -------- SETTER: CHANGE XAI TYPE --------
    def set_generation_method(self, method, model):
        """
        Change the method for generating attention/heat-maps.
        Args:
            method: "cdam" or "gradcam".
            model: Swin V2 model.
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
        Set the Grad-CAM method.
        """
        # Get the target layer for Grad-CAM (using the projection of the last attention block)
        target_layer = model.swin.layers[-1].blocks[-1].attn
        self.grad_cam = LayerGradCam(model, target_layer.proj)
        self.generate_attention_map = self.__generate_attention_map_grad_cam

    def __set_cdam(self, model):
        """
        Set the CDAM method.
        """
        self.cdam = LayerAttribution(model, model.swin.layers[-1].blocks[-1].attn.proj)
        self.generate_attention_map = self.__extract_attention_cdam

    # ============== GRAD-CAM =================
    def __generate_attention_map_grad_cam(self, model, image_tensor):
        """
        Generate an attention map using Grad-CAM.
        Args:
            model: Swin V2 model.
            image_tensor: Pre-processed image tensor (e.g., shape (1, 1, 256, 256)).
        Returns:
            Normalized heatmap (numpy array) in the range [0, 255].
        """
        model.eval()
        # Compute Grad-CAM attribution
        attribution = self.grad_cam.attribute(image_tensor, target=0)
        heatmap = attribution.squeeze().cpu().detach().numpy()

        # Cast to float32 for compatibility with OpenCV
        # Uncomment if needed
        # heatmap = heatmap.astype(np.float32)

        heatmap_norm = np.zeros_like(heatmap)  # Create an empty array for normalization with same shape
        res_heatmap = cv2.normalize(heatmap_norm, heatmap, 0.0, 255.0, cv2.NORM_MINMAX) # Normalize in place
        self.map = res_heatmap
        return res_heatmap

    # ============== CDAM =================
    def __extract_attention_cdam(self, model, image_tensor, layer_num=-1):
        """
        Extract attention weights from a specific layer of Swin V2 (CDAM).
        Args:
            model: Pre-trained Swin V2 model.
            image_tensor: Pre-processed image tensor (e.g., shape (1, 1, 256, 256)).
            layer_num: Transformer layer index (-1 for last layer).
        Returns:
            Normalized attention heatmap.
        """
        model.eval()
        # Retrieve attention weights using a custom method (assumes get_attn() exists)
        transformer_layer = model.swin.layers[layer_num].blocks[-1].attn.get_attn()
        with torch.no_grad():
            _ = model(image_tensor)
        attention_map = transformer_layer.cpu().detach().numpy()
        # Average over all heads
        attention_map = np.mean(attention_map, axis=0)
        # Resize to match desired output size
        attention_map = cv2.resize(attention_map, (256, 256))
        # Normalize the attention map to [0, 255]
        res_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)
        self.map = res_map
        return res_map

    # ========= ANALYSIS =============
    def analyze_intensity(self, image, is_3d=False):
        """
        Analyze pixel intensity within the attention map region.
        Args:
            image: Grayscale X-ray image.
            is_3d: If True, collapse the heatmap to 2D using mean.
        Returns:
            Tuple: (mean_intensity, variance_intensity) of the image region indicated by the attention map.
        Raises:
            ValueError: If no attention map has been generated.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        if is_3d:
            # Collapse across the first dimension if needed
            heatmap = np.mean(self.map, axis=0)
            heatmap = heatmap / np.max(heatmap)
        else:
            heatmap = self.map / np.max(self.map)

        # Create a mask by selecting pixels with high attention
        self.mask = heatmap > ATTENTION_MAP_THRESHOLD  # Threshold

        # Extract the intensity values from the original image
        intensities = image[self.mask]

        # Calculate mean and variance of the intensities
        mean_intensity = np.mean(intensities)
        variance_intensity = np.var(intensities)
        return mean_intensity, variance_intensity

    # ======== VISUALIZATION ============
    def visualize_attention(self, image_tensor):
        """
        Overlay the attention map on the original image.
        Args:
            image_tensor: Pre-processed image tensor (e.g., shape (1, 1, 256, 256)).
        Raises:
            ValueError: If no attention map has been generated.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        # Convert image tensor to numpy array
        image = image_tensor.squeeze().cpu().detach().numpy()
        # If image is 2D (grayscale), convert to 3 channels for display purposes.
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        else:
            # Assume image tensor in (C, H, W) format; transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))

        # Resize the attention map to the original image dimensions
        attention_map = cv2.resize(self.map, (image.shape[1], image.shape[0]))
        # If image values are in [0,1], scale the attention map accordingly
        if np.max(image) <= 1.0:
            attention_map = attention_map / 255.0
        else:
            # Convert attention map to 3 channels
            attention_map = cv2.cvtColor(attention_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Ensure both images are in uint8 format
        image_uint8 = (image * 255).astype(np.uint8) if np.max(image) <= 1.0 else image.astype(np.uint8)
        overlay = cv2.addWeighted(image_uint8, 0.5, attention_map.astype(np.uint8), 0.5, 0)
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    def visualize_attention_map_only(self):
        """
        Display only the attention map.
        Raises:
            ValueError: If no attention map has been generated.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")
        plt.imshow(self.map, cmap='jet')
        plt.colorbar()
        plt.show()

    # ========== ADJUSTMENT =============
    def adjust_attention_map(self, mean_intensity, pathology):
        """
        Adjust the attention map based on clinical consistency.
        Args:
            mean_intensity: Mean intensity of the attention map region.
            pathology: Pathology type.
        Returns:
            The adjusted attention map.
        Raises:
            ValueError: If no attention map has been generated.
        """
        if self.map is None:
            raise ValueError("No attention map generated yet.")

        # Compare with clinical values (this is a placeholder for more sophisticated logic)
        if self.compare_with_clinical_values(mean_intensity, np.var(self.map), pathology) == "❌ Potential inconsistency":
            # Reduce attention map values in the region of interest
            self.map[self.map > ATTENTION_MAP_THRESHOLD] *= 0.7
        return self.map

    def compare_with_clinical_values(self, mean_intensity, variance_intensity, pathology):
        """
        Compare intensity values with known clinical ranges.
        This is a placeholder function; extend as needed.
        """
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
