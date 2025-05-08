import io
import os
import random
import time

import gcsfs
import torch
import torchvision.transforms as transforms
from gcsfs.retry import HttpError
from google.api_core.exceptions import TooManyRequests
from torch.utils.data import Dataset
from PIL import Image

from settings import BILLING_PROJECT, SERVICE_ACCOUNT_TOKEN, BUCKET_PREFIX_PATH, DATASET_PATH
import threading

# Lock for thread-safe access to gcsfs client
_FS_LOCK = threading.Lock()
# Cache for gcsfs clients
_FS_CACHE = {}  # one gcsfs client per billing project


# ====================== HELPER FUNCTIONS ======================= #

def pil_cloud_open(path):
    """
    Open an image file and convert it to the specified mode.
    Returns a PIL.Image *from local file or gs:// object*.
    If `billing_project` env variable (see settings.py) is supplied, requester-pays header is added.

    Args:
        path (str): Path to the image file.
    Returns:
        PIL.Image: Image object in the specified mode, from local file or gs:// object.

    """
    billing_project = BILLING_PROJECT
    requester_pays = True if billing_project else False
    token = SERVICE_ACCOUNT_TOKEN

    if billing_project is None:
        raise ValueError(
            "[ERROR]: BILLING_PROJECT is not set. Please set it in your environment variables. - Not implemented yet")

    fs = _FS_CACHE.get(billing_project)
    with _FS_LOCK:
        if fs is None:
            fs = gcsfs.GCSFileSystem(
                project=billing_project or "auto",
                requester_pays=requester_pays,
                token=token
            )
            _FS_CACHE[billing_project] = fs

    for attempt in range(3):
        try:
            with fs.open(path, "rb") as f:
                buf = io.BytesIO(f.read())
            return Image.open(buf)

        except (IOError, HttpError, TooManyRequests, FileNotFoundError):
            if attempt == 2:  # If it's the last attempt, skip this item
                print(f"Failed to preprocess image {path} after 3 attempts. Skipping.")
                raise IndexError(f"Skipping due to repeated failures.")
            time.sleep((2 ** attempt) + random.random())


# Function to resize while maintaining aspect ratio and add padding
def preprocess_image(image, channels_mode="L", image_size=(256, 256), view_position='AP'):
    """
    Preprocess the image by resizing it while maintaining the aspect ratio and adding padding.
    The View Position is  'AP' or 'PA' for Anterior-Posterior or Posterior-Anterior projection.
    If 'PA', the image will be flipped horizontally.

    Args:
        image(PIL.Image.Image): The image to be preprocessed.
        channels_mode (str): Color mode of the image. Default is "L". Accepts "RGB" or "L" (grayscale).
        image_size (tuple[int, int]): Desired output size of the image. Default is (256, 256).
        view_position (str): The view position of the image. Default is 'AP'.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    img = image
    img = img.convert(channels_mode)  # Convert to grayscale if needed
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
        transforms.Normalize(mean=mean, std=std),  # Standard Normalization,
        transforms.Lambda(lambda x: x.flip(2) if view_position == 'PA' else x)  # Flip horizontally if 'PA'
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
    def __init__(self, data_dict, transform = None, image_size=(256, 256),
                 channels_mode="L", return_study_id=False, use_bucket=False):
        """
        Initialize the dataset with image paths, labels and transformations.
        `data_dict` structure:
            Each key should be an index containing a dict for each image.
            Structure of dictionary:
            {idx: {
                "path": image_path,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "subject_id": subject_id,
                "label": [label1, label2, ...],
                "view_position": view_position
            }}
        Parameters:
            data_dict (dict): Dictionary with all
            transform (callable, optional): Optional transform to be applied on a sample (already opened image).
            image_size (tuple): Desired output size of the image.
            channels_mode (str): Color mode of the image. Default is "RGB". Accepts "RGB" or "L" (grayscale).
            return_study_id (bool): If True, the dataloader will return the study_id along with the image and label in the tuple.
            use_bucket (bool): If True, the function will use the bucketed dataset in Dataloader.
            It defaults to False in order to avoid unnecessary checks.

        """

        self.image_size = image_size
        self.data_dict = data_dict


        for _, img_dict in data_dict.items():
            prefix = "gs://" + BUCKET_PREFIX_PATH if use_bucket else DATASET_PATH
            img_dict["path"] = os.path.join(prefix, img_dict["path"])
        self.use_bucket = use_bucket

        if channels_mode not in ["RGB", "L"]:
            raise ValueError("channels_mode must be either 'RGB' or 'L'")
        self.channels_mode = channels_mode
        self.transform = transform

        self.return_study_id = return_study_id

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to be fetched.
        Returns:
            tuple: (torch.Tensor, torch.Tensor) where [0] is the preprocessed image tensor
            and [1] is the corresponding label.
            If return_study_id is True (in init), it also returns the study_id tensor.
        """
        img_pth = self.data_dict[idx]["path"]
        # Use the cloud open function if "use_bucket" is True
        # Else open the image using PIL
        img = pil_cloud_open(img_pth) if self.use_bucket else Image.open(img_pth)

        # Convert labels to Tensor
        label_tensor = torch.tensor(self.data_dict[idx]['labels'],
                                    dtype=torch.float)

        # Use the transform defined in the constructor
        # If no transform is provided, use the default preprocessing function
        res_list = [self.transform(img) if self.transform
                    else preprocess_image(img,
                                          channels_mode = self.channels_mode,
                                          view_position= self.data_dict[idx]["view_position"]
                                          ),
                    label_tensor] # Append the label tensor to the result list

        # If return_study_id is True, extract the study_id
        if self.return_study_id:
            study_id = self.data_dict[idx]["study_id"]
            study_id_tensor = torch.tensor([int(study_id)], dtype=torch.float)
            res_list.append(study_id_tensor)

        return tuple(res_list)


# =============================================