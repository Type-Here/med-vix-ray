import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


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

    # Creation of a white image (padding) (black in this case)
    padded_img = Image.new("RGB", (256, 256), (0, 0, 0))
    padded_img.paste(img, ((256 - new_width) // 2, (256 - new_height) // 2))

    # Convert to tensor and normalize for PyTorch
    transform = transforms.Compose([
        #transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standard Normalization
    ])

    return transform(padded_img).unsqueeze(0)  # Adds batch dimension


class ImagePreprocessor(Dataset):
    """
        Custom dataset class for loading and preprocessing images.
        Args:
            image_paths (list): List of paths to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_size (tuple): Desired output size of the image.
    """
    def __init__(self, image_paths, transform = None, image_size=(256, 256)):
        self.image_size = image_size
        self.image_paths = image_paths

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to be fetched.
        Returns:
            Tensor: Preprocessed image tensor.
        """
        img_pth = self.image_paths[idx]
        if self.transform:
            # use the transform defined in the constructor
            return self.transform(img_pth)
        else:
            return preprocess_image(img_pth)

# =============================================


if __name__ == "__main__":
    # Example usage
    img_path = "/path/to/sample_image.jpg"  # Replace with your image path
    image_tensor = preprocess_image(img_path)
    print(f"Shape immagine pre-processata: {image_tensor.shape}")  # Output: torch.Size([1, 3, 256, 256])
