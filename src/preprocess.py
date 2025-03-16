import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


# Function to resize while maintaining aspect ratio and add padding
def preprocess_image(image_path, channels_mode="RGB", image_size=(256, 256)):
    """
    Preprocess the image by resizing it while maintaining the aspect ratio and adding padding.
    :param image_path: Path to the image file.
    :param channels_mode: RGB or L (grayscale). Default is "RGB".
    :param image_size: Size of the output image. Default is (256, 256). #
    :return:
    """
    img = Image.open(image_path).convert(channels_mode)  # Convert to Specified Channel for ViT compatibility

    # Resizing while maintaining aspect ratio
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:  # Image wider than tall
        new_width = image_size[0]
        new_height = int(image_size[1] / aspect_ratio)
    else:  # Image taller than wide
        new_height = image_size[1]
        new_width = int(image_size[0] * aspect_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if channels_mode == "L":
        color = 0
        mean = 0.5
        std = 0.5
    else:
        color = (255, 255, 255)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    # Creation of a white image (padding) (black in this case)
    padded_img = Image.new(channels_mode, image_size, color=color)
    padded_img.paste(img, ((image_size[0] - new_width) // 2, (image_size[1] - new_height) // 2))

    # Convert to tensor and normalize for PyTorch
    transform = transforms.Compose([
        #transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Standard Normalization
    ])

    #return transform(padded_img).unsqueeze(0)  # Adds batch dimension
    return transform(padded_img)  # DataLoader will add batch dimension


class ImagePreprocessor(Dataset):
    """
        Custom dataset class for loading and preprocessing images.
        This class is used to load images from the specified paths,
        apply transformations and return the preprocessed images along with their labels.

        Parameters:
            image_paths (list): List of paths to the images.
            image_labels (dict): Dictionary with image name for key, and list of labels for value.
            Should be already aligned with the image_paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_size (tuple): Desired output size of the image.
            channels_mode (str): Color mode of the image. Default is "RGB". Accepts "RGB" or "L" (grayscale).
        :returns: tuple(torch.Tensor, torch.Tensor): Preprocessed image tensor; List of labels for the image.
        :rtype: tuple
    """
    def __init__(self, image_paths, image_labels, transform = None, image_size=(256, 256), channels_mode="RGB"):

        self.image_size = image_size
        self.image_labels = image_labels
        self.image_paths = image_paths

        if channels_mode not in ["RGB", "L"]:
            raise ValueError("channels_mode must be either 'RGB' or 'L'")
        self.channels_mode = channels_mode
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to be fetched.
        Returns:
            tuple: (torch.Tensor, torch.Tensor) where [0] is the preprocessed image tensor and [1] is the corresponding label.
        """
        img_pth = self.image_paths[idx]
        key = img_pth.split("/")[-1].split(".")[0] # Suppose image key is dicom_id from mimic
        # Convert labels to Tensor
        label_tensor = torch.tensor(self.image_labels[key], dtype=torch.float)
        if self.transform:
            # use the transform defined in the constructor
            return self.transform(img_pth), label_tensor
        else:
            return preprocess_image(img_pth, channels_mode=self.channels_mode), label_tensor

# =============================================


if __name__ == "__main__":
    # Example usage
    img_path = "/path/to/sample_image.jpg"  # Replace with your image path
    image_tensor = preprocess_image(img_path)
    print(f"Shape immagine pre-processata: {image_tensor.shape}")  # Output: torch.Size([1, 3, 256, 256])
