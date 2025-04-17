import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from settings import DATASET_PATH, DATASET_INFO_CSV_DIR, TRAIN_TEST_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, \
    SPLITTED_DATASET_DIR, MIMIC_LABELS, IMAGES_SET_PATHS_AVAILABLE

labels = MIMIC_LABELS  # List of labels for the dataset

"""
    Dataset handling functions for loading, splitting, and processing the dataset.
    These functions are used to prepare the dataset for training and evaluation.
    The dataset is expected to be in a specific directory structure and format.
    Manage also the labels and metadata and partial list of images.
"""

def _load_dataset_metadata():
    """
    Load dataset metadata from the specified directory.

    Returns:
        pd.DataFrame: DataFrame containing dataset metadata.
    Raises:
        FileNotFoundError: If the dataset path does not exist.
    """
    # Check if the dataset path exists
    if not os.path.exists(DATASET_INFO_CSV_DIR):
        raise FileNotFoundError(f"Dataset path {DATASET_INFO_CSV_DIR} does not exist.")

    # Load the dataset metadata
    metadata_path = os.path.join(DATASET_INFO_CSV_DIR, 'metadata.csv')
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    else:
        raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")


def _load_labels():
    """
    Load labels from the specified directory.
    Returns:
        pd.DataFrame: DataFrame containing labels.
    Raises:
        FileNotFoundError: If the dataset path or file does not exist.
    """
    # Check if the dataset path exists
    if not os.path.exists(DATASET_INFO_CSV_DIR):
        raise FileNotFoundError(f"Dataset path {DATASET_PATH} does not exist.")

    # Load the labels
    labels_path = os.path.join(DATASET_INFO_CSV_DIR, 'labels.csv')
    if os.path.exists(labels_path):
        return pd.read_csv(labels_path)
    else:
        raise FileNotFoundError(f"Labels file {labels_path} does not exist.")


def load_ready_dataset(phase='train'):
    """
    Load the dataset for the specified phase (train, validation, test).

    Args:
        phase (str): Phase of the dataset to load. Can be 'train', 'validation', or 'test'.

    Returns:
        pd.DataFrame: DataFrame containing the dataset for the specified phase.

    Raises:
        FileNotFoundError: If the dataset path or file does not exist.
        ValueError: If the phase is not one of 'train', 'validation', or 'test'.
    """
    if phase not in ['train', 'validation', 'test']:
        raise ValueError("Phase must be 'train', 'validation', or 'test'.")

    # Check if the dataset path exists
    if not os.path.exists(DATASET_INFO_CSV_DIR):
        raise FileNotFoundError(f"Dataset path {DATASET_INFO_CSV_DIR} does not exist.")

    # Load the dataset based on the phase
    dataset_path = os.path.join(SPLITTED_DATASET_DIR, f'{phase}_data.csv')
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")


def dataset_handle(partial_list=None):
    """
    Handle the dataset by loading metadata and labels. \n
    If partial_list is provided, drop all rows that are not in the partial_list. \n
    It expects a file formatted as:
    Each line: files/patient_starting_id/p"subject_id"/s"study_id"/"dicom_id".jpg \n
    Example: files/p10/p1000001/s50000001/xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx.jpg
    Args:
        partial_list (str, optional): Path to a txt file containing a list of rows to keep or available.
        If None, all rows are kept. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the merged dataset with labels.
    """
    # Load metadata and labels
    metadata = _load_dataset_metadata()
    labels_csv = _load_labels()

    # Merge metadata and labels on the 'study_id' column
    merged_data = pd.merge(metadata, labels_csv, on='study_id', how='inner')

    # Keep only AP or PA views (ViewPosition column)
    merged_data = merged_data[merged_data['ViewPosition'].isin(['AP', 'PA'])]

    # Print the number of records in the merged data
    print(f"Number of records in merged data: {len(merged_data)}")

    # Labels in csv are:
    # 0.0: Negative
    # 1.0: Positive
    # -1.0: Uncertain
    # NaN: Uncertain
    # Iterate through the labels and replace -1.0 and NaN with 0.0
    for label in labels: # labels list above
        merged_data[label] = merged_data[label].replace(-1.0, 0.0)
        merged_data[label] = merged_data[label].fillna(0.0)

    if partial_list:
        # Load the partial list
        with open(partial_list, 'r') as f:
            partials = f.read().splitlines()
        if not partials:
            raise ValueError("Partial list is empty. Please provide a valid file.")

        # Extract the DICOM IDs from the partial list
        partials = [line.split('/')[-1].replace('.jpg', '') for line in partials]

        # Filter the merged data based on the partial list
        merged_data = merged_data[merged_data['dicom_id'].isin(partials)]

        # Close the file
        f.close()

    # Print Number of AP and PA views
    print(f"Number of AP views: {len(merged_data[merged_data['ViewPosition'] == 'AP'])}")
    print(f"Number of PA views: {len(merged_data[merged_data['ViewPosition'] == 'PA'])}")

    return merged_data


def split_dataset(merged_data, train_ratio=TRAIN_TEST_SPLIT, val_ratio=VALIDATION_SPLIT, test_ratio=TEST_SPLIT):
    """
    Split the dataset into training, validation, and test sets.
    The split is done in a stratified manner based on the labels.

    Args:
        merged_data (pd.DataFrame): DataFrame containing the merged dataset.
        train_ratio (float): Ratio of training data. Default in settings.py is 0.8
        val_ratio (float): Ratio of validation data. Default in settings.py is 0.1
        test_ratio (float): Ratio of test data. Default in settings.py is 0.1

    Returns:
        tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame): Tuple containing the training, validation, and test sets.
    """
    # Check if the ratios sum to 1
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Ratios must sum to 1.")

    # -- Not Used Too Raw
    # Shuffle the dataset
    # shuffled_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # -- Not Used Too Much Classes for 14 labels: lots of unique combinations
    # Create a column that concatenates the label values into a binary string
    #merged_data['label_combination'] = merged_data[labels].apply(lambda row: ''.join(row.astype(str)), axis=1)

    # Create a column that concatenates the label values into a binary string
    merged_data['num_labels'] = merged_data[labels].apply(lambda row: (row == 1.0).sum(), axis=1)

    # Remove rows with NaN in labels to avoid errors
    merged_data = merged_data.dropna(subset=['num_labels'])

    # First split: 80% for training, 20% for validation + test (stratified on the number of labels)
    train_data, temp_data = train_test_split(
        merged_data,
        test_size=0.2,
        stratify=merged_data['num_labels'], # Stratify on the number of labels
        random_state=42
    )

    # Print the distribution after the split to check the balance
    print("Distribuzione nel Training Set:\n", train_data['num_labels'].value_counts(normalize=True))
    print("\nDistribuzione nel Test Set:\n", temp_data['num_labels'].value_counts(normalize=True))

    # Second split: 50% for validation, 50% for test (stratified on the label combination)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5,
                                                  stratify=temp_data['num_labels'], # Stratify on the number of labels
                                                  random_state=42)

    # Save the splits to CSV files
    split_dir = SPLITTED_DATASET_DIR
    os.makedirs(split_dir, exist_ok=True)
    train_data.to_csv(os.path.join(split_dir, 'train_data.csv'), index=False)
    validation_data.to_csv(os.path.join(split_dir, 'validation_data.csv'), index=False)
    test_data.to_csv(os.path.join(split_dir, 'test_data.csv'), index=False)

    return train_data, validation_data, test_data


def _build_image_index(image_dir, save_path=IMAGES_SET_PATHS_AVAILABLE):
    """
    Scans the image directory and builds a set of all .jpg image paths available on disk.
    It also saves the set to a file for future reference.

    Args:
        image_dir (str): Root directory containing the image data.

    Returns:
        set: Set of full image paths found on disk (for fast lookup).
    """
    print(f"[INFO] Scanning '{image_dir}' for image files...")

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg"):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    print(f"[INFO] Found {len(image_paths)} images on disk.")
    set_paths = set(image_paths)

    # Save the set to a file for future reference as .pkl
    with open(save_path, 'wb') as f:
        pickle.dump(set_paths, f)

    return set_paths


def fetch_image_from_csv(csv_file, image_dir=DATASET_PATH, image_index=None):
    """
    Fetch images from the dataset based on the CSV file.

    Args:
        csv_file (str | pd.DataFrame | PathLike): Path to the CSV file containing image paths or pd.DataFrame.
        image_dir (str): Main Parent directory where the images are stored.
        image_index (set, optional): Set of valid image paths to filter. If None, will be built automatically.
    Returns:
        list: List of image paths.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If csv_file is neither a DataFrame nor a string.
    """
    if not os.path.exists(image_dir):
        print(f"No valid DATASET_PATH: {image_dir} found. Check Environment variable.")
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    # Check if csv_file is a DataFrame or a string
    if isinstance(csv_file, pd.DataFrame):
        # If it's a DataFrame, use it directly
        df = csv_file
    elif isinstance(csv_file, str):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} does not exist.")
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
    else:
        raise ValueError("csv_file must be a DataFrame or a string path to a CSV file.")

    # Build image index if not provided
    if image_index is None and not os.path.exists(IMAGES_SET_PATHS_AVAILABLE):
        # Build the image index from the image directory
        print(f"[INFO] Image index not found. Building from {image_dir}...")
        image_index = _build_image_index(image_dir)

    elif image_index is None:
        try:
            with open(IMAGES_SET_PATHS_AVAILABLE, 'rb') as f:
                image_index = pickle.load(f)
            print(f"[INFO] Loaded image index set from {IMAGES_SET_PATHS_AVAILABLE}.")
        except Exception as e:
            print(f"[WARNING] Failed to load cached image index: {e}")
            print(f"[INFO] Rebuilding image index from {image_dir}...")
            image_index = _build_image_index(image_dir)

    # Create a list to store the image paths
    image_paths = []

    # Check if subject_id column is subject_id or subject_id_x (from splitting)
    if 'subject_id_x' in df.columns:
        subject_id_col = 'subject_id_x'
    else:
        subject_id_col = 'subject_id'

    for _, row in df.iterrows():
        study_id = row['study_id']
        subject_id = str(row[subject_id_col])
        dicom_id = row['dicom_id']

        folder_path = os.path.join(f"p{subject_id[0:2]}", f"p{subject_id}", f"s{study_id}")
        image_path = os.path.join(image_dir, folder_path, dicom_id + '.jpg')

        if image_path in image_index:
            image_paths.append(image_path)
        else:
            # Optional: limit noisy logs
            print(f"[SKIP] Missing image: {image_path}")
            continue

    print(f"[INFO] Found {len(image_paths)} valid images out of {len(df)} records.")
    return image_paths