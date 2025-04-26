import json, os
from settings import NER_GROUND_TRUTH

if __name__ == "__main__":
    print("Starting...")
    # Retrieve directory of ground truth NER results
    ground_truth_dir = os.path.dirname(NER_GROUND_TRUTH)

    all_results = {}

    for ner in os.listdir(ground_truth_dir):
        if ner.endswith(".json"):
            print(f" -- Joining Report {ner}... -- ")

            ner_path = os.path.join(ground_truth_dir, ner)  # Construct full path
            # Load the NER results
            with open(ner_path, 'r') as file:
                ner_data = json.load(file)

            # Adding the results to all_results
            all_results.update(ner_data)

    print("Joining completed.")
    # print statistics of the all_results
    print(f"Total number of results: {len(all_results)}")

    # Create a json file with all the results
    print("Saving file with all results grouped...")
    with open('../report_keywords/all_comparison_results.json', 'w') as file:
        json.dump(all_results, file, indent=4)
    print("All results saved to all_comparison_results.json")

    print("Done. Exiting...")