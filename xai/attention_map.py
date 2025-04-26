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
        self.generate_attention_map = self.__generate_attention_map_cdam

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

    def __generate_attention_map_cdam(self, model, image_tensor, layer_num=-1):
        """
        Modified to support batch input.
        If image_tensor has shape [B, C, H, W], this function returns a tensor of shape [B, H_out, W_out].
        Args:
            model: Pre-trained Swin V2 model.
            image_tensor: Pre-processed image tensor (e.g., shape (B, 1, 256, 256)).
            layer_num: Transformer layer index (-1 for last layer).
        Returns:
            Normalized attention heatmap (torch tensor).
        """
        b_size = image_tensor.shape[0]
        att_maps = []
        for i in range(b_size):
            # Process each image separately using the existing function.
            single_att_map = self.__extract_attention_cdam_single_image(model, image_tensor[i:i + 1], layer_num)
            # Convert the resulting numpy array to a torch tensor.
            att_maps.append(torch.tensor(single_att_map, dtype=torch.float32, device=image_tensor.device))
        return torch.stack(att_maps, dim=0)

    def __extract_attention_cdam_single_image(self, model, image_tensor, layer_num=-1):
        """
        Extract attention weights from a specific layer of Swin V2 (CDAM).
        Note:
            On model.layers[layer_num].blocks[-1].attn:
            It expects attn_weights attribute to be present,
            if it doesn't exist yet, ensure to create a hook in init of the model.
        Args:
            model: Pre-trained Swin V2 model.
            image_tensor: Pre-processed image tensor (e.g., shape (1, 1, 256, 256)).
            layer_num: Transformer layer index (-1 for last layer).
        Returns:
            Normalized attention heatmap.
        """
        # 1. Save the current mode and set the model to evaluation mode.
        prev_mode = model.training

        # Ensure the model is in evaluation mode
        model.eval()

        # 2. Run a forward pass in no_grad mode to ensure attention weights are computed.
        with torch.no_grad():
            _ = model(image_tensor)

        # 3. Get the attention weights from the specified layer.
        # See Method Note for info
        attn = model.layers[layer_num].blocks[-1].attn
        transformer_layer = attn.attn_weights
        #print("Layer Obtained:", transformer_layer)

        # 4. Restore the model's original training mode.
        if prev_mode:
            model.train()

        # 5. Convert attention weights to a NumPy array and average over heads.
        attention_map = transformer_layer.cpu().detach().numpy()
        # Standard Swin V2: [1, 64, 1024]
        shape = attention_map.shape

        # Remove batch and head
        # shape: [1, 1, 64, 1024] → [64, 1024]
        if len(shape) == 4:
            _, num_heads, tokens, num_windows = shape
            attention_map = attention_map[0, 0]
        elif len(shape) == 3:
            num_heads, tokens, num_windows = shape
            attention_map = attention_map[0]
        else:
            raise ValueError(f"Invalid attention map shape: {shape}")

        # Average over all heads
        attention_map = np.mean(attention_map, axis=0) # TODO Check if mean is the best option

        # 6. Reconstruct grid of windows [32, 32]
        win_grid = int(np.sqrt(num_windows))
        assert win_grid * win_grid == num_windows, f"Non quadrato: {num_windows}"
        att_map = attention_map.reshape((win_grid, win_grid))  # [32, 32]

        # 7. Use the input image dimensions for resizing.
        # Resize to match desired output size
        # Assume image_tensor shape is (B, C, H, W); use dimensions of the first image.
        _, _, hei, wi = image_tensor.shape
        new_size = (int(wi), int(hei))
        att_map_resized = cv2.resize(att_map, new_size, interpolation=cv2.INTER_CUBIC)

        # 8. Normalize the attention map to the range [0, 1].
        map_dst = np.zeros_like(att_map_resized, dtype=np.float32)
        res_map = cv2.normalize(att_map_resized, map_dst, 0, 1, cv2.NORM_MINMAX)

        # Do not store the result in self.map to avoid side effects in concurrent calls.
        # self.map = res_map
        return res_map

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
