import os

from torch.utils.data import DataLoader

import dataset.dataset_handle as dh
from settings import DOWNLOADED_FILES, DATASET_PATH, MIMIC_LABELS, NUM_WORKERS
from src.preprocess import ImagePreprocessor

"""
    General utility functions for the project model.
"""


def menu():
    """
    Display the main menu for the project.
    """
    print("Welcome to the MIMIC-CXR Project!")
    print("1. Baseline Evaluation")
    print("2. Fine Tuned Model Train or Evaluation")
    print("2. Graph Model (Med-ViX-Ray Train or Evaluation")
    print("9. Exit")
    choice = input("Please select an option (1-3): ")
    return choice


def model_option(model_path, model_obj):
    """
    Check if a model exists at the specified path. If it does, prompt the user to load it or train a new one.
    If the model does not exist, prompt the user to train a new one.
    Args:
        model_path (str): Path to the model.
        model_obj (nn.Module): Model object to load the model into.
    :return:
    """
    if os.path.exists(model_path):
        print(f"Model exists in {model_path}; Load it?")
        if input("y/N: ").lower() == "y":
            model_obj.load_model(model_path)
            print("Model loaded.")
            print("Model:", model_obj, "\nSwin Part:", model_obj.swin_model)
            return False

        elif input("Do you want to train a new model? (y/N): ").lower() == "y":
            return True
        else:
            print("Exiting...")
            exit(0)
    else:
        print("Model not found. Training a new model.")
        return True

# ======= STANDARD OPERATIONS TO GET TRAINING AND VALIDATION DATASET AND LABELS =======

def _load_train_val_sets():
    """
    Load the training and validation datasets.
    If they do not exist, create them using dataset_handle.load_ready_dataset function.
    Returns:
        tuple(pd.DataFrame, pd.DataFrame): The training and validation datasets.
    """
    # Load dataset
    try:
        train_dataset = dh.load_ready_dataset(phase='train')
    except FileNotFoundError:
        print("Train dataset not found. Creating a new one.")
        merged_data = dh.dataset_handle(partial_list=DOWNLOADED_FILES)
        train_dataset, _, _ = dh.split_dataset(merged_data)

    try:
        validation_dataset = dh.load_ready_dataset(phase='validation')
    except FileNotFoundError:
        print("Validation dataset not found. Since Train dataset was created or loaded \n"
              " there should be a validation dataset too.")
        print("Check the dataset folder or code.")
        exit(1)

    print("Train and Validation information dataset loaded.")
    return train_dataset, validation_dataset


def _get_image_paths_from_csv(train_dataset=None, validation_dataset=None):
    """
    Fetch image paths from the CSV files for training and validation datasets.
    Note: if the dataset is not provided, it will return None in the tuple
    Args:
        train_dataset (pd.DataFrame): The training dataset.
        validation_dataset (pd.DataFrame): The validation dataset.
    Returns:
        tuple(list, list): The training and validation image paths.
    """
    train_image_paths, validation_image_paths = None, None
    if train_dataset is not None:
        train_image_paths = dh.fetch_image_from_csv(train_dataset, DATASET_PATH)
    if validation_dataset is not None:
        validation_image_paths = dh.fetch_image_from_csv(validation_dataset, DATASET_PATH)

    return train_image_paths, validation_image_paths


def _get_train_val_labels(train_dataset=None, validation_dataset=None):
    """
    Fetch labels from pd.DataFrame for training and validation datasets.
    Note: if the dataset is not provided, it will return None in the tuple
    Args:
        train_dataset (pd.DataFrame): The training dataset.
        validation_dataset (pd.DataFrame): The validation dataset.
    Returns:
        tuple(dict, dict): The training and validation labels.
    """
    train_labels, val_labels = None, None
    # Convert labels to dictionary
    if train_dataset is not None:
        train_labels = {train_dataset['dicom_id'][i]: train_dataset[MIMIC_LABELS].iloc[i].tolist()
                        for i in range(len(train_dataset))}
    if validation_dataset is not None:
        val_labels = {validation_dataset['dicom_id'][i]: validation_dataset[MIMIC_LABELS].iloc[i].tolist()
                      for i in range(len(validation_dataset))}
    return train_labels, val_labels


# =============================== MAIN CALLING FUNCTION TO GET DIRECTLY THE DATALOADERS ===============================


def get_dataloaders(return_study_id=False, pin_memory=False):
    """
        Get the training and validation dataloaders.

        This function loads the datasets, fetches the image paths and labels and creates the dataloaders.
        Datasets are the same for every model, using settings.py variables to get the paths.

        The dataloaders are created using the ImagePreprocessor class for preprocessing images.
        Args:
            return_study_id (bool): If True, the dataloader will return the study_id
            along with the image and label in the tuple.
            pin_memory (bool): If True, the dataloader will use pinned memory for faster data transfer to GPU.

        Returns:
            tuple: (DataLoader, DataLoader) for training and validation datasets.
    """
    # Load the dataset with partially downloaded files
    train_dataset, validation_dataset = _load_train_val_sets()
    print("Train and Validation datasets loaded.")

    # Obtain Paths
    train_image_paths, val_image_paths = _get_image_paths_from_csv(train_dataset, validation_dataset)
    if len(train_image_paths) == 0 or len(val_image_paths) == 0:
        print("No images found in the dataset. Check the dataset folder or code.")
        exit(1)

    # Obtain Labels
    train_labels, val_labels = _get_train_val_labels(train_dataset, validation_dataset)
    print("Train and Validation labels loaded.")

    # Obtain Dataloaders in order to improve performance
    training_loader = DataLoader(ImagePreprocessor(train_image_paths, train_labels,
                                                   channels_mode="L", return_study_id=return_study_id),
                                 batch_size=16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    valid_loader = DataLoader(ImagePreprocessor(val_image_paths, val_labels,
                                                channels_mode="L", return_study_id=return_study_id),
                              batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    return training_loader, valid_loader