import os

from torch.utils.data import DataLoader

import dataset.dataset_handle as dh
from settings import NUM_WORKERS, BATCH_SIZE
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


def basic_menu_model_option(model_path, model_obj):
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


# =============================== MAIN CALLING FUNCTION TO GET DIRECTLY THE DATALOADERS ===============================


def get_dataloaders(return_study_id=False, pin_memory=False,
                    return_train_loader=True, return_val_loader=True,
                    full_data=False, verify_existence=False, use_bucket=False, channels_mode="RGB"):
    """
        Get the training and validation dataloaders.

        This function loads the datasets, fetches the image paths and labels and creates the dataloaders.
        Datasets are the same for every model, using settings.py variables to get the paths.

        Note:

        -If return_train_loader or return_val_loader is set to False,
        the function will return None for that dataloader but a tuple is always returned.

        -When all_data is set to True, the function will load the entire dataset using mimic Physionet division csv info.
        The directory for the dataset is set to MIMIC_SPLIT_DIR in settings.py.

        -The dataloaders are created using the ImagePreprocessor class for preprocessing images.
        Args:
            return_study_id (bool): If True, the dataloader will return the study_id
            along with the image and label in the tuple.

            pin_memory (bool): If True, the dataloader will use pinned memory for faster data transfer to GPU.

            return_train_loader (bool): If True, the dataloader will return the training dataloader.

            return_val_loader (bool): If True, the dataloader will return the validation dataloader.

            full_data (bool): If True, the function will load the entire dataset using mimic physionet division csv info.

            use_bucket (bool): If True, the function will use the bucketed dataset in Dataloader.
            It will change the image_dir path to use a FUSE mounted bucket.

            verify_existence (bool): If True, the function will check if the image paths
            exist while fetching paths from csv. (Only for non bucketed dataset)
            
            channels_mode (str): Color mode of the image. Default is "RGB". Accepts "RGB" or "L" (grayscale).

        Returns:
            tuple[DataLoader, DataLoader] for training and validation datasets.
    """
    if use_bucket and not full_data:
        print("[WARNING] Using bucketed dataset with partial data.")

    if use_bucket and verify_existence:
        print("[WARNING] Using bucketed dataset with verify_existence=True. "
              "This will not work as the image paths are not verified in the bucket.")
        verify_existence = False

    print("\n ------------------- ")
    print("[INFO] Loading dataset(s) metadata...")

    training_loader, valid_loader = None, None
    if return_train_loader:
        print("[INFO] Fetching training metadata...")
        train_metadata = dh.fetch_metadata(phase='train',
                                           full_data=full_data,
                                           verify_existence=verify_existence)
        if len(train_metadata) == 0:
            print("[ERROR] No images found in the dataset for Training. "
                  "Check the dataset folder or code.")
            exit(1)

        print(" - Train metadata size:", len(train_metadata))
        print(" - Creating training dataloader...")
        training_loader = DataLoader(ImagePreprocessor
                                     (train_metadata,
                                      channels_mode=channels_mode,
                                      return_study_id=return_study_id,
                                      use_bucket=use_bucket),
                                 batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=pin_memory)

    if return_val_loader:
        print("[INFO] Fetching validation metadata...")
        val_metadata = dh.fetch_metadata(phase='val',
                                         full_data=full_data,
                                         verify_existence=verify_existence)
        if len(val_metadata) == 0:
            print("[ERROR] No images found in the dataset for Validation. "
                  "Check the dataset folder or code.")
            exit(1)

        print(" - Validation metadata size:", len(val_metadata))
        print(" - Creating validation dataloader...")
        valid_loader = DataLoader(ImagePreprocessor
                                  (val_metadata,
                                   channels_mode=channels_mode,
                                   return_study_id=False,
                                   use_bucket=use_bucket),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=pin_memory)

    return training_loader, valid_loader


def get_test_dataloader( pin_memory=False,
                         full_data=False, verify_existence=False, use_bucket=False, channels_mode="RGB"):
    """
        Get the test dataloader.

        This function loads the test dataset, fetches the image paths and labels and creates the dataloader.
        The dataset is the same for every model, using settings.py variables to get the paths.
        Channels mode is present to allow for RGB or grayscale images so also baseline models can be used.
        Args:
            channels_mode (str): Color mode of the image. Default is "RGB". Accepts "RGB" or "L" (grayscale).

            pin_memory (bool): If True, the dataloader will use pinned memory for faster data transfer to GPU.

            full_data (bool): If True, the function will load the entire dataset using mimic physionet division csv info.

            use_bucket (bool): If True, the function will use the bucketed dataset in Dataloader.
            It will change the image_dir path to use a FUSE mounted bucket.

            verify_existence (bool): If True, the function will check if the image paths
            exist while fetching paths from csv. (Only for non bucketed dataset)

        Returns:
            DataLoader for test dataset.
    """
    if use_bucket and not full_data:
        print("[WARNING] Using bucketed dataset with partial data.")

    if use_bucket and verify_existence:
        print("[WARNING] Using bucketed dataset with verify_existence=True. "
              "This will not work as the image paths are not verified in the bucket.")
        verify_existence = False

    print("\n ------------------- ")
    print("[INFO] Loading test dataset metadata...")

    print("[INFO] Fetching test metadata...")
    test_metadata = dh.fetch_metadata(phase='test',
                                      full_data=full_data,
                                      verify_existence=verify_existence)
    if len(test_metadata) == 0:
        print("[ERROR] No images found in the dataset for Testing. "
              "Check the dataset folder or code.")
        exit(1)

    print(" - Test metadata size:", len(test_metadata))
    print(" - Creating test dataloader...")
    test_loader = DataLoader(ImagePreprocessor
                             (test_metadata,
                              channels_mode=channels_mode,
                              return_study_id=False,
                              use_bucket=use_bucket),
                             batch_size=1, shuffle=False,
                             num_workers=1,
                             pin_memory=pin_memory)
    return test_loader