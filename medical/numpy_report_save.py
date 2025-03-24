import os
import numpy as np
import concurrent.futures

from settings import MIMIC_REPORT_DIR


def process_directory(directory):
    """
    Processes a single directory by reading all .txt files within it.

    Args:
        directory (str): Path to the directory.

    Returns:
        dict: A dictionary mapping file base names (without extension) to file contents.
    """
    reports = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    report_id = os.path.splitext(file)[0]
                    reports[report_id] = text
                except Exception as e:
                    print(f"Error processing file {path}: {e}")
    return reports


def collect_reports_txt(directories, max_workers=3):
    """
    Processes multiple directories concurrently, each in its own thread.

    Args:
        directories (list): List of directory paths.
        max_workers (int): Maximum number of threads to use concurrently.

    Returns:
        dict: Combined dictionary of all report texts from all directories.
    """
    combined_reports = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each directory to be processed in a separate thread.
        futures = {executor.submit(process_directory, directory): directory for directory in directories}
        total = len(futures)
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            progress = (completed / total) * 100
            print(f"Progress: {progress:.2f}%")
            directory = futures[future]
            try:
                reports = future.result()
                combined_reports.update(reports)
            except Exception as exc:
                print(f"Directory {directory} generated an exception: {exc}")
    return combined_reports


def collect_reports_txt_single_thread(directories):
    """
    Walks through given directories and collects all .txt files.

    Args:
        directories (list): List of directory paths to search.

    Returns:
        dict: A dictionary with report IDs (derived from file names) as keys
              and the file content as values.
    """
    reports = {}
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    # Use file name without extension as the report ID
                    report_id = os.path.splitext(file)[0]
                    reports[report_id] = text
    return reports

if __name__ == "__main__":
    read = True
    if read:
        # Load the reports from the .npy file
        try:
            reports_dict = np.load("../ontology/scispacy/nlp_tests/reports.npy", allow_pickle=True).item()
            print("Reports loaded successfully.")

            # Print the number of reports loaded
            print(f"Number of reports loaded: {len(reports_dict)}")
            # Print the first 5 report IDs
            print("First 5 report IDs:", list(reports_dict.keys())[:5])
            # Print the first 5 report contents
            print("First 5 report contents:", list(reports_dict.values())[:5])

        except FileNotFoundError:
            print("No reports.npy file found. Please run the script to generate it.")
            exit(1)

    else:
        # Find all directories with pXX (where X is a digit) inside the MIMIC_REPORT_DIR
        # New function to list all files (excluding subdirectories) with full paths in a given directory
        def list_files_in_directory(directory):
            return [os.path.join(directory, entry) for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry)) and entry.startswith("p")]

        dirs = list_files_in_directory(MIMIC_REPORT_DIR)

        # Collect the reports into a dictionary
        reports_dict = collect_reports_txt(dirs, max_workers=4)

        # Save the dictionary as a .npy file
        np.save("../ontology/scispacy/nlp_tests/reports.npy", reports_dict)

        print("Reports saved as reports.npy")
