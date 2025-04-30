import torch
import cv2  # For image processing, resize, and normalization
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as fc

from captum.attr import LayerGradCam, LayerAttribution
from settings import ATTENTION_MAP_THRESHOLD


class AttentionMap:
    def __init__(self, model, xai_type="cdam"):
        """
        Initialize the AttentionMap class.
        This class is used to generate attention maps using either CDAM or Grad-CAM methods.
        Note:
            - Grad-CAM does not support batch input.
            - CDAM supports batch input.
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
            print("Using CDAM...")
            self.__set_cdam(model)
        else:
            print("Using Grad-CAM for attention map generation.")
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
        Warning:
            Grad-CAM does not support batch input!
        """
        # Get the target layer for Grad-CAM (using the projection of the last attention block)
        target_layer = model.layers[-1].blocks[-1].attn
        self.grad_cam = LayerGradCam(model, target_layer.proj)
        self.generate_attention_map = self.__generate_attention_map_grad_cam

    def __set_cdam(self, model):
        """
        Set the CDAM method.
        Note:
            The CDAM method supports batch input also!
        """
        self.cdam = LayerAttribution(model, model.layers[-1].blocks[-1].attn.proj)
        self.generate_attention_map = self.__extract_attention_batch

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

    @torch.no_grad()
    def __extract_attention_batch(self, model, images, layer_idx=-1, target_size=(256, 256)):
        """
        Extract attention maps for a whole batch efficiently.

        Args:
            model (torch.nn.Module): Your model with attention hooks.
            images (torch.Tensor): Input batch [B, C, H, W].
            layer_idx (int): Transformer layer to extract from.
            target_size (tuple): Size to resize the attention maps.

        Returns:
            torch.Tensor: Attention maps resized [B, 1, H, W].
        """
        model.eval()

        # Get Batch size from input
        batch = images.shape[0]

        # Forward pass
        _ = model(images)

        # Retrieve attention weights
        attn = model.layers[layer_idx].blocks[-1].attn.attn_weights  # [B, H, N, N]

        if attn.dim() == 4:
            # [B,H,Nw,Np] -> [B,Nw,Np]
            attn_map = attn.mean(dim=1)
        elif attn.dim() == 3:
            # [H,Nw,Np] -> [Nw,Np], poi replicare su B
            tmp = attn.mean(dim=0)  # [Nw,Np]
            attn_map = tmp.unsqueeze(0).repeat(batch, 1, 1)  # [B,Nw,Np]
        else:
            raise ValueError(f"Unexpected attn shape: {attn.shape}")

        # 2. Use mean across rows to summarize per token
        # Take attention to the class token if needed
        # For Swin no cls_token, so we can average
        #    [B,Nw,Np] -> [B,Np]
        attn_vec = attn_map.mean(dim=1)  # [B, Np]

        # 3. reshape in mappa quadrata
        batch_, n_p = attn_vec.shape
        side = int(n_p ** 0.5)
        if side * side != n_p:
            raise ValueError(f"Cannot reshape vector of length {n_p} into square (got side={side}).")
        attn_map2d = attn_vec.view(batch_, 1, side, side)

        # 4 .Resize to match input image size
        attn_map_resized = fc.interpolate(attn_map2d, size=target_size, mode='bilinear', align_corners=False)

        # 5. Normalize each map individually [0,1]
        flat = attn_map_resized.view(batch_, -1)
        mn = flat.min(dim=1, keepdim=True)[0]
        mx = flat.max(dim=1, keepdim=True)[0]
        norm = (flat - mn) / (mx - mn + 1e-6)

        return norm.view(batch_, 1, *target_size)

    # ========= ANALYSIS =============
    def analyze_intensity(self, image, is_3d=False):
        """
        Analyze pixel intensity within the attention map region.
        Args:
            image (np.array): Grayscale X-ray image as a numpy array.
            is_3d (bool): If True, collapse the heatmap to 2D using mean.
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
        # returns a boolean array
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
