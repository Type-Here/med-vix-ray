import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from settings import DATASET_PATH, DATASET_INFO_CSV_DIR, TRAIN_TEST_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, \
    MIMIC_LABELS, MIMIC_SPLIT_CSV, DOWNLOADED_FILES, CSV_METADATA_DIR, PICKLE_METADATA_DIR

labels = MIMIC_LABELS  # List of labels for the dataset

_DATA_CACHE = {}

"""
    Dataset handling functions for loading, splitting, and processing the dataset.
    These functions are used to prepare the dataset for training and evaluation.
    The dataset is expected to be in a specific directory structure and format.
    Manage also the labels and metadata and partial list of images.
"""


def __save_split_datasets(directory, train_data, validation_data, test_data, full_data=False):
    """
    Save the split datasets to the specified directory.

    Args:
        directory (str): Directory where the split datasets will be saved.
        train_data (pd.DataFrame): Training dataset.
        validation_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Test dataset.
        full_data (bool): If True, the csv contains the full dataset metadata.
        Otherwise, it contains only the partial dataset and '_partial_data' suffix is added to the csv name.
    """
    saving_names = ['train_data', 'validation_data', 'test_data']
    if not full_data:
        # Append to csv name '_partial_data'
        saving_names = [name + '_partial_data' for name in saving_names]

    os.makedirs(directory, exist_ok=True)
    train_data.to_csv(os.path.join(directory, saving_names[0] + '.csv'), index=False)
    validation_data.to_csv(os.path.join(directory, saving_names[1] + '.csv'), index=False)
    test_data.to_csv(os.path.join(directory, saving_names[2] + '.csv'), index=False)

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
    Load labels from the specified directory
    Returns:
        pd.DataFrame: DataFrame containing labels.
    Raises:
        FileNotFoundError: If the dataset path or file does not exist.
    """
    # Check if the dataset path exists
    if not os.path.exists(DATASET_INFO_CSV_DIR):
        raise FileNotFoundError(f"Labels path directory {DATASET_INFO_CSV_DIR} does not exist.")

    # Load the labels
    labels_path = os.path.join(DATASET_INFO_CSV_DIR, 'labels.csv')
    if os.path.exists(labels_path):
        return pd.read_csv(labels_path)
    else:
        raise FileNotFoundError(f"Labels file {labels_path} does not exist.")


def __load_ready_dataset(csv_name, directory=CSV_METADATA_DIR):
    """
    Load the dataset for the specified phase (train, validation, test).

    Args:
        csv_name (str): The Simple-name of the CSV file containing the dataset. (Not the full path)
        directory (str): Directory where the split datasets are stored.
    Returns:
        pd.DataFrame: DataFrame containing the dataset for the specified phase.

    Raises:
        FileNotFoundError: If the dataset path or file does not exist.
        ValueError: If the phase is not one of 'train', 'validation', or 'test'.
    """
    # Load the dataset based on the phase
    dataset_path = os.path.join(directory, csv_name)

    if not os.path.exists(dataset_path):
        return None
    print(f" -- [INFO] Loading csv metadata from {dataset_path}...")
    return pd.read_csv(dataset_path)


def __generate_merged_metadata(partial_list=None):
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
    # Load metadata and labels -- Contain ALL ROWS data
    metadata = _load_dataset_metadata()
    labels_csv = _load_labels()

    # Merge metadata and labels on the 'study_id' column
    merged_data = pd.merge(metadata, labels_csv, on='study_id', how='inner', suffixes=('_metadata', '_labels'))

    # Check for _metadata and _labels duplicate columns
    merged_data = __resolve_duplicated_columns_csv(merged_data, metadata, labels_csv, suffixes=('_metadata', '_labels'))

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

    # Add to cache
    key = 'merged_data' if not partial_list else 'merged_data_partial_data'
    _DATA_CACHE[key] = merged_data

    return merged_data


def __resolve_duplicated_columns_csv(merged_data, x_df, y_df, suffixes=('_x', '_y')):
    """
        Auxiliary function to resolve duplicates in the merged dataset.
        It checks for columns with the same name in the merged dataset and
        automatically resolves conflicts by keeping the values from one of the columns.
        Args:
            merged_data (pd.DataFrame): DataFrame containing the merged dataset.
            x_df (pd.DataFrame): First DataFrame merged.
            y_df (pd.DataFrame): Second DataFrame merged.
            suffixes (tuple): Suffixes added to the column names in case of duplicates.
        Returns:
            pd.DataFrame: DataFrame with resolved duplicates.
    """
    # Automatically resolve conflicts for columns with the same name
    for col in x_df.columns.intersection(y_df.columns):
        col_x = col + suffixes[0]
        col_y = col + suffixes[1]
        if col_x in merged_data.columns and col_y in merged_data.columns:
            # Check for conflicts and handle them
            conflicts = merged_data[merged_data[col_x] != merged_data[col_y]]
            if not conflicts.empty:
                print(f"[WARNING] Conflicts found in column '{col}': {len(conflicts)} rows")
                print(f"[INFO] Keeping column '{col_x}' values.")

            col_name = col
            merged_data[col] = merged_data[col_x]
            # Drop the duplicate columns
            merged_data.drop(columns=[col_x, col_y], inplace=True)
    return merged_data

def __split_dataset(merged_data, train_ratio=TRAIN_TEST_SPLIT,
                    val_ratio=VALIDATION_SPLIT, test_ratio=TEST_SPLIT, partial_list=None):
    """
    Split the dataset into training, validation, and test sets.
    The split is done in a stratified manner based on the labels.

    Args:
        merged_data (pd.DataFrame): DataFrame containing the merged dataset.
        train_ratio (float): Ratio of training data. Default in settings.py is 0.8
        val_ratio (float): Ratio of validation data. Default in settings.py is 0.1
        test_ratio (float): Ratio of test data. Default in settings.py is 0.1
        partial_list (str): Path to a txt file containing a list of rows to keep or available.
    Returns:
        tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame): Tuple containing the training, validation, and test sets.
    Raises:
        ValueError: If the ratios do not sum to 1 or if the partial list is None.
    """
    if partial_list is None:
        print("[ERROR] No partial list provided. If you want to use the full dataset: \n "
              "Use the MIMIC split function '__split_dataset_using_mimic_split' instead.")
        raise ValueError("Partial list is None. Please provide a valid file.")

    # Check if the ratios sum to approximately 1 (within a small tolerance)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
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
    print("-Distribuzione nel Training Set:\n", train_data['num_labels'].value_counts(normalize=True))
    print("\n-Distribuzione nel Val + Test Set:\n", temp_data['num_labels'].value_counts(normalize=True))

    # Second split: 50% for validation, 50% for test (stratified on the label combination)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5,
                                                  stratify=temp_data['num_labels'], # Stratify on the number of labels
                                                  random_state=42)
    # Print the distribution after the split to check the balance
    print("\n-Distribuzione nel Validation Set:\n", validation_data['num_labels'].value_counts(normalize=True))
    print("\n-Distribuzione nel Test Set:\n", test_data['num_labels'].value_counts(normalize=True))

    # Save the splits to CSV files
    __save_split_datasets(CSV_METADATA_DIR, train_data, validation_data, test_data, full_data=False)

    return train_data, validation_data, test_data


def __split_dataset_using_mimic_split(merged_data, csv_split_info=MIMIC_SPLIT_CSV):
    """
    Split the dataset into training, validation, and test sets using the MIMIC split.
    The split is done in a stratified manner based on the labels.

    Args:
        merged_data (pd.DataFrame): DataFrame containing the merged dataset.
        csv_split_info (str): Path to the CSV file containing the MIMIC split information.
    Returns:
        tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame): Tuple containing the training, validation, and test sets.
    """
    # Load the MIMIC split
    if not os.path.exists(csv_split_info):
        raise FileNotFoundError(f"MIMIC split file {csv_split_info} does not exist.")

    mimic_split = pd.read_csv(csv_split_info)

    # Merge the MIMIC split with the merged data
    new_merged_data = pd.merge(merged_data, mimic_split, on='dicom_id', how='inner')
    # Check for duplicated columns in the merged data
    merged_data = __resolve_duplicated_columns_csv(new_merged_data, mimic_split, merged_data)

    # Split the dataset based on the MIMIC split
    train_data = merged_data[merged_data['split'] == 'train']
    validation_data = merged_data[merged_data['split'] == 'validate']
    test_data = merged_data[merged_data['split'] == 'test']

    # Save the splits to CSV files
    __save_split_datasets(CSV_METADATA_DIR, train_data,
                              validation_data, test_data, full_data=True)

    return train_data, validation_data, test_data


def __compose_image_path(single_metadata_dict):
    """
        Given a single metadata dictionary, generate the full path to the image.
        Args:
            single_metadata_dict (dict): Dictionary containing metadata for a single image.
        Returns:
            str: Full path to the image.
    """

    study_id = single_metadata_dict['study_id']
    subject_id = single_metadata_dict['subject_id']
    dicom_id = single_metadata_dict['dicom_id']

    subfolder_path = os.path.join(f"p{subject_id[0:2]}", f"p{subject_id}", f"s{study_id}")
    image_path = os.path.join(subfolder_path, dicom_id + '.jpg')
    return image_path


def fetch_metadata(phase=None, full_data=False, verify_existence=False):
    """
    Fetch metadata from the dataset.
    Args:
        phase (str): Type of metadata to fetch. Can be 'train', 'val', or 'test'.
        full_data (bool): If True, return the full dataset. If False, return only the specified type.
        verify_existence (bool): If True, verify the existence of the images in the dataset.

    Returns:
        dict: Dictionary containing the metadata requested.
        It contains a key idx for each image.
        Each key contains a dictionary with the metadata:
            'dicom_id', 'labels', 'study_id', 'view_position', 'subject_id', 'path'.
    """
    if phase not in ['train', 'val', 'test']:
        raise ValueError("Type must be 'train', 'val', 'test'")

    file_name = f'{phase}_metadata.pkl' if full_data else f'{phase}_partial_data.pkl'
    if not full_data:
        # Append to pickle name '_partial_data'
        file_name = file_name.replace('.pkl', '_partial_data.pkl')
    if verify_existence:
        # Append to pickle name '_verify_existence'
        file_name = file_name.replace('.pkl', '_verified.pkl')

    pickle_path = os.path.join(PICKLE_METADATA_DIR, file_name)
    os.makedirs(PICKLE_METADATA_DIR, exist_ok=True)
    # If pickle file is found, return it
    if os.path.exists(pickle_path):
        print(f" - [INFO] Loading metadata from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    # If not found create it
    return _fetch_metadata_from_csv(phase, full_data=full_data,
                                        verify_existence=verify_existence)


def _fetch_metadata_from_csv(phase, full_data=False, verify_existence=False):
    """
    Fetch metadata from the dataset.
    Args:
        phase (str): Type of metadata to fetch. Can be 'train', 'val', or 'test'.
        full_data (bool): If True, return the full dataset.
        If False, return only the specified type.
        verify_existence (bool): If True, verify the existence of the images in local disk.
    Returns:
        dict: Dictionary containing the metadata requested.
        It contains a dictionary for each image (keys are indexes).
        Each key contains a dictionary with at least the following metadata:
         'dicom_id', 'labels', 'study_id', 'view_position', 'subject_id', 'path'.
    """

    # Check if CSV of split dataset is available
    if phase not in ['train', 'val', 'test']:
        raise ValueError("Phase must be 'train', 'validation', or 'test'.")

    csv_name = f'{phase}_data.csv' if full_data else f'{phase}_partial_data.csv'

    # Retrieve the dataset from the CSV file
    dataset = _load_csv_data(phase, csv_name, full_data=full_data)

    if DATASET_PATH is None:
        raise ValueError("DATASET_PATH is not set. Please set it in settings.py.")

    # Create a dictionary to store the metadata
    metadata = []
    for _, row in dataset.iterrows():
        entry = {
            'dicom_id': str(row['dicom_id']),
            'labels': row[MIMIC_LABELS].tolist(),
            'study_id': str(row['study_id']),
            'view_position': row['ViewPosition'],
            'subject_id': str(row['subject_id']),
        }
        entry['path'] = __compose_image_path(entry)
        metadata.append(entry)

    # Save the metadata to a pickle file
    pickle_name = f'{phase}_metadata.pkl' if full_data else f'{phase}_partial_data.pkl'

    # Check if the images exist in the dataset directory
    if verify_existence:
        # Append to pickle name '_verify_existence'
        metadata = __verify_existence(metadata, dataset_dir = DATASET_PATH)
        pickle_name = pickle_name.replace('.pkl', '_verified.pkl')

    # Save the metadata to a pickle file
    pickle_path = os.path.join(PICKLE_METADATA_DIR, pickle_name)
    with open(pickle_path, 'wb') as f:
        print(f" - [INFO] Saving metadata to {pickle_path}...")
        pickle.dump(metadata, f)

    return metadata


def _load_csv_data(phase, csv_name, full_data):
    """
        This function loads the dataset info
        for a specific phase (train, validation, test) from a CSV file.

        It checks if the file exists in the specified directory.
        If it does, it loads the dataset from the CSV file.
        If it doesn't, it generates the dataset by splitting the original dataset.
        All Split datasets are then saved to CSV files for future use.
        Args:
            phase (str): Phase of the dataset to load. Can be 'train', 'validation', or 'test'.
            csv_name (str): Name of the CSV file to load.
            full_data (bool): If True, the csv contains the full dataset metadata.
            Otherwise, it contains only the partial dataset and '_partial_data' suffix is added to the csv name.
        Returns:
            pd.DataFrame: DataFrame containing the dataset for the specified phase.
    """

    # Try to load the dataset from the CSV file
    dataset = __load_ready_dataset(csv_name, directory=CSV_METADATA_DIR)
    if dataset is not None:
        print(f" - [INFO] Found Split Dataset from {csv_name}...")
        return dataset

    # Else, generate the dataset
    print(f" - [INFO] No already split dataset found."
          f" Fetching from the original dataset or merged dataset.")

    merged_data_name = 'merged_data' if not full_data else 'merged_data_partial_data'
    merged_data = _DATA_CACHE.get(merged_data_name, None)
    # If merged data is not in cache, generate it
    if merged_data is None:
        # Load the merged dataset
        print(" - [INFO] Creating merged dataset...")
        merged_data = __generate_merged_metadata()

    if full_data:
        train_data, validation_data, test_data = __split_dataset_using_mimic_split(merged_data)
    else:
        train_data, validation_data, test_data = __split_dataset(merged_data, DOWNLOADED_FILES)

    print(" - [INFO] Generated the split datasets and saved them to CSV files.")

    # Return only the requested dataset
    if phase == 'train':
        return train_data
    elif phase == 'val':
        return validation_data
    elif phase == 'test':
        return test_data
    else:
        raise ValueError("Phase must be 'train', 'validation', or 'test'.")


def __verify_existence(metadata, dataset_dir):
    """
        Verify the existence of the images in the dataset directory.
        Args:
            metadata (list): List containing the metadata as dictionaries.
            dataset_dir (str): Directory where the images are stored.
        Returns:
            dict: Dictionary containing the metadata with verified image paths.
    """
    print(f" - [INFO] Verifying the existence of images in {dataset_dir}...")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"[ERROR] Dataset directory {dataset_dir} does not exist.")

    for i, data in enumerate(metadata):
        if not os.path.exists(os.path.join(dataset_dir, data['path'])):
            print(f"[-] Image {data['path']} does not exist. Removing from metadata.")
            # Remove the image from the metadata
            del metadata[i]
    return metadata
