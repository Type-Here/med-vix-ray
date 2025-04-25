import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from settings import DATASET_PATH, DATASET_INFO_CSV_DIR, TRAIN_TEST_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, \
    SPLIT_DATASET_DIR, MIMIC_LABELS, IMAGES_SET_PATHS_AVAILABLE, MIMIC_SPLIT_CSV, MIMIC_SPLIT_DIR

labels = MIMIC_LABELS  # List of labels for the dataset

"""
    Dataset handling functions for loading, splitting, and processing the dataset.
    These functions are used to prepare the dataset for training and evaluation.
    The dataset is expected to be in a specific directory structure and format.
    Manage also the labels and metadata and partial list of images.
"""

def __save_split_datasets(directory, train_data, validation_data, test_data):
    """
    Save the split datasets to the specified directory.

    Args:
        directory (str): Directory where the split datasets will be saved.
        train_data (pd.DataFrame): Training dataset.
        validation_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Test dataset.
    """
    os.makedirs(directory, exist_ok=True)
    train_data.to_csv(os.path.join(directory, 'train_data.csv'), index=False)
    validation_data.to_csv(os.path.join(directory, 'validation_data.csv'), index=False)
    test_data.to_csv(os.path.join(directory, 'test_data.csv'), index=False)

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


def load_ready_dataset(phase='train', directory=SPLIT_DATASET_DIR):
    """
    Load the dataset for the specified phase (train, validation, test).

    Args:
        phase (str): Phase of the dataset to load. Can be 'train', 'validation', or 'test'.
        directory (str): Directory where the split datasets are stored.
    Returns:
        pd.DataFrame: DataFrame containing the dataset for the specified phase.

    Raises:
        FileNotFoundError: If the dataset path or file does not exist.
        ValueError: If the phase is not one of 'train', 'validation', or 'test'.
    """
    if phase not in ['train', 'validation', 'test']:
        raise ValueError("Phase must be 'train', 'validation', or 'test'.")

    # Load the dataset based on the phase
    dataset_path = os.path.join(directory, f'{phase}_data.csv')

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
    # Load metadata and labels -- Contain ALL ROWS data
    metadata = _load_dataset_metadata()
    labels_csv = _load_labels()

    # Merge metadata and labels on the 'study_id' column
    merged_data = pd.merge(metadata, labels_csv, on='study_id', how='inner', suffixes=('_metadata', '_labels'))

    # Automatically resolve conflicts for columns with the same name
    for col in metadata.columns.intersection(labels_csv.columns):
        if col + '_metadata' in merged_data.columns and col + '_labels' in merged_data.columns:
            # Check for conflicts and handle them
            conflicts = merged_data[merged_data[col + '_metadata'] != merged_data[col + '_labels']]
            if not conflicts.empty:
                print(f"[WARNING] Conflicts found in column '{col}': {len(conflicts)} rows")
                print("[INFO] Keeping metadata values.")

            merged_data[col] = merged_data[col + '_metadata']
            # Drop the duplicate columns
            merged_data.drop(columns=[col + '_metadata', col + '_labels'], inplace=True)

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


def split_dataset(merged_data, train_ratio=TRAIN_TEST_SPLIT,
                  val_ratio=VALIDATION_SPLIT, test_ratio=TEST_SPLIT, partial_list=None):
    """
    Split the dataset into training, validation, and test sets.
    The split is done in a stratified manner based on the labels.

    Args:
        merged_data (pd.DataFrame): DataFrame containing the merged dataset.
        train_ratio (float): Ratio of training data. Default in settings.py is 0.8
        val_ratio (float): Ratio of validation data. Default in settings.py is 0.1
        test_ratio (float): Ratio of test data. Default in settings.py is 0.1
        partial_list (str, optional): Path to a txt file containing a list of rows to keep or available.
        If None: all data will be managed from mimic split info.

    Returns:
        tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame): Tuple containing the training, validation, and test sets.
    """
    if partial_list is None:
        return split_dataset_using_mimic_split(merged_data)

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
    split_dir = SPLIT_DATASET_DIR
    __save_split_datasets(split_dir, train_data, validation_data, test_data)

    return train_data, validation_data, test_data


def split_dataset_using_mimic_split(merged_data):
    """
    Split the dataset into training, validation, and test sets using the MIMIC split.
    The split is done in a stratified manner based on the labels.

    Args:
        merged_data (pd.DataFrame): DataFrame containing the merged dataset.

    Returns:
        tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame): Tuple containing the training, validation, and test sets.
    """
    # Load the MIMIC split
    if not os.path.exists(MIMIC_SPLIT_CSV):
        raise FileNotFoundError(f"MIMIC split file {MIMIC_SPLIT_CSV} does not exist.")

    mimic_split = pd.read_csv(MIMIC_SPLIT_CSV)

    # Merge the MIMIC split with the merged data
    merged_data = pd.merge(merged_data, mimic_split, on='dicom_id', how='inner')

    # Split the dataset based on the MIMIC split
    train_data = merged_data[merged_data['split'] == 'train']
    validation_data = merged_data[merged_data['split'] == 'val']
    test_data = merged_data[merged_data['split'] == 'test']

    if not os.path.exists(MIMIC_SPLIT_DIR):
        __save_split_datasets(MIMIC_SPLIT_DIR, train_data,
                              validation_data, test_data)

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


def fetch_image_from_csv(csv_file, image_dir_prefix=DATASET_PATH, csv_kind='train', use_csv_data_only=False):
    """
    Fetch images from the dataset based on the CSV file.

    Args:
        csv_file (str | pd.DataFrame | PathLike): Path to the CSV file containing image paths or pd.DataFrame.
        image_dir_prefix (str): Main Parent directory where the images are stored.
        csv_kind (str): Kind of CSV file. Can be 'train', 'validation', or 'test'.
        This is used to retrieve the correct pickle file.
        use_csv_data_only (bool): If True, only use the images available in the CSV file
        to build the index saved in the pickle file.
    Returns:
        list: List of image paths.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If csv_file is neither a DataFrame nor a string.
    """
    if not os.path.exists(image_dir_prefix):
        print(f"[WARNING]: No valid DATASET_PATH: {image_dir_prefix} found. Check Environment variable.")
        # raise FileNotFoundError(f"Image directory {image_dir_prefix} does not exist.")

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

    # Create a list to store the image paths
    image_paths = []

   # Rename subject_id_x and study_id_x columns to remove the '_x' suffix if they exist
    if 'subject_id_x' in df.columns:
        df.rename(columns={'subject_id_x': 'subject_id'}, inplace=True)
    if 'study_id_x' in df.columns:
        df.rename(columns={'study_id_x': 'study_id'}, inplace=True)

    # Modify the pickle file path to include the csv_kind
    image_index = IMAGES_SET_PATHS_AVAILABLE.split('.pkl')[0] + f'_{csv_kind}.pkl'

    # Build image index if not provided
    if not os.path.exists(image_index) or use_csv_data_only:
        print(f"[INFO] Image index not found or use_csv_data_only is True. Building image index...")
        return __build_image_index_and_fetch_from_csv(image_paths, df,
                                                          subject_id_col="subject_id",
                                                          image_dir_prefix=image_dir_prefix)
    try:
        with open(IMAGES_SET_PATHS_AVAILABLE, 'rb') as f:
            image_index = pickle.load(f)
        print(f"[INFO] Loaded image index set from {IMAGES_SET_PATHS_AVAILABLE}.")
    except Exception as e:
        print(f"[WARNING] Failed to load cached image index: {e}")
        print(f"[INFO] Rebuilding image index from {image_dir_prefix}...")
        image_index = _build_image_index(image_dir_prefix)

    return __fetch_image_paths_only(csv_file, image_dir_prefix, image_paths,
                                    image_index, df, subject_id_col="subject_id")


def __fetch_image_paths_only(csv_file, image_dir_prefix, image_paths, image_index,
                             df, subject_id_col='subject_id'):
    """
    Fetch image paths from the CSV file and check if they exist in the image index already built.
    Args:
        csv_file (str): Path to the CSV file containing image paths.
        image_dir_prefix (str): Main Parent directory where the images are stored.
        image_paths (list): List to stored image paths.
        image_index (set): Set of image paths available on disk.
        df (pd.DataFrame): DataFrame containing the metadata.
        subject_id_col (str): Column name for subject ID in the DataFrame.
    Returns:
        list: List of image paths to use.
    """
    for _, row in df.iterrows():
        study_id = row['study_id']
        subject_id = str(row[subject_id_col])
        dicom_id = row['dicom_id']

        subfolder_path = os.path.join(f"p{subject_id[0:2]}", f"p{subject_id}", f"s{study_id}")
        image_path = os.path.join(image_dir_prefix, subfolder_path, dicom_id + '.jpg')

        if image_path in image_index:
            image_paths.append(image_path)
        else:
            # Optional: limit noisy logs
            print(f"[SKIP] Missing image: {image_path}")
            continue

    print(f"[INFO] Found {len(image_paths)} valid images out of {len(df)} records for csv: {csv_file}.")
    return image_paths


def __build_image_index_and_fetch_from_csv(image_paths, dataframe,
                                           subject_id_col='subject_id', image_dir_prefix=DATASET_PATH, save=True):
    """
        Build a list of image paths from the DataFrame and save them to a list using pickle.
        It will check only for the images available in the CSV file and not all the images in the working directory.
        Args:
            image_paths (list): List to store the image paths.
            dataframe (pd.DataFrame): DataFrame containing the metadata.
            subject_id_col (str): Column name for subject ID in the DataFrame.
            image_dir_prefix (str): Main Parent directory where the images are stored.
        Note:
            It will save the available images in the image_paths list as pkl file checking only the csv data
            instead of all working directory.
        Returns:
            list: List of image paths.
    """
    # Iterate through the DataFrame and construct the full image paths
    for index, row in dataframe.iterrows():
        # Extract folder path
        study_id = row['study_id']
        subject_id = str(row[subject_id_col])

        # Construct the image path
        folder_path = os.path.join(f"p{subject_id[0:2]}", f"p{subject_id}", f"s{study_id}")
        dicom_id = row['dicom_id']

        # Construct the full image path
        image_path = os.path.join(image_dir_prefix, folder_path, dicom_id + '.jpg')
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"[SKIP] - Missing image: {image_path}")

    # Saving the image paths to a pickle file
    if save:
        with open(IMAGES_SET_PATHS_AVAILABLE, 'wb') as f:
            print("Saving the image paths to a pickle file as set.")
            set_paths = set(image_paths)
            pickle.dump(set_paths, f)

    return image_paths