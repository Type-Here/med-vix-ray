import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# Function to resize while maintaining aspect ratio and add padding
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert to RGB for ViT compatibility

    # Resizing while maintaining aspect ratio
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:  # Image wider than tall
        new_width = 256
        new_height = int(256 / aspect_ratio)
    else:  # Image taller than wide
        new_height = 256
        new_width = int(256 * aspect_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Creation of a white image (padding)
    padded_img = Image.new("RGB", (256, 256), (0, 0, 0))
    padded_img.paste(img, ((256 - new_width) // 2, (256 - new_height) // 2))

    # Convert to tensor and normalize for PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standard Normalization
    ])

    return transform(padded_img).unsqueeze(0)  # Adds batch dimension

# =============================================


if __name__ == "__main__":
    # Example usage
    img_path = "/path/to/sample_image.jpg"  # Replace with your image path
    image_tensor = preprocess_image(img_path)
    print(f"Shape immagine pre-processata: {image_tensor.shape}")  # Output: torch.Size([1, 3, 256, 256])
