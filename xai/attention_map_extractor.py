import torch
from timm.models.swin_transformer_v2 import window_reverse

class SelfAttentionMapExtractor:
    """
    Extracts interpretable attention maps from a SwinV2 model using the final attention scores.
    This is a rollout-like method, not CDAM or GradCAM.
    """
    def __init__(self, model, target_layer=None):
        """
        Args:
            model (nn.Module): The SwinV2 model.
            target_layer (str or None): Name of the attention layer to extract. If None, uses last block.
        """
        self.model = model
        self.device = next(model.parameters()).device

        # Default to last block attention
        if target_layer is None:
            self.target_attn = model.layers[-1].blocks[-1].attn
        else:
            self.target_attn = eval(f"model.{target_layer}")  # risky if not trusted input

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention map from input x.
        Args:
            x (Tensor): [B, C, H, W]
        Returns:
            Tensor: Attention maps of shape [B, H_out, W_out]
        """
        # Ensure model in eval mode and disable grad
        was_training = self.model.training
        b_size, _, h_img, w_img = x.shape

        self.model.eval()

        with torch.no_grad():
            _ = self.model(x)

        # Restore training mode if it was on
        if was_training:
            self.model.train()

        attn_scores = self.target_attn.attn_weights  # [B, H, N, N]
        if attn_scores is None:
            raise ValueError("Attention weights not captured. Check hook.")

        attn_scores = attn_scores.mean(dim=1)  # mean over heads → [B, N, N]
        diag_attn = attn_scores.diagonal(dim1=-2, dim2=-1)  # self-attention

        bat, n_tok = diag_attn.shape
        # Reshape to square grid
        win_size = int(n_tok ** 0.5)

        # Reshape the diagonal attention scores into a square grid
        diag_attn = diag_attn.view(bat, win_size, win_size)  # [B, H, W]

        # Convert to global attention map using window_reverse [B, H_feat, W_feat]
        attn_global = window_reverse(diag_attn, (win_size, win_size), (h_img, w_img))

        # Resize to original image size
        attn_resized = torch.nn.functional.interpolate(attn_global, size=(h_img, w_img),
                                                       mode='bicubic', align_corners=False)

        # Normalize per sample between [0, 1]
        attn_min = attn_resized.amin(dim=(2, 3), keepdim=True)
        attn_max = attn_resized.amax(dim=(2, 3), keepdim=True)
        attn_norm = (attn_resized - attn_min) / (attn_max - attn_min + 1e-6)

        # Final shape: [B, H, W]
        return attn_norm.squeeze(1)