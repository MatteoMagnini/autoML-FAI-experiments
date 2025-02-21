import os
import numpy as np
import pandas as pd

def read_results(args: dict) -> (dict, list):
    def get_dict_from_pattern(pattern: str = None) -> dict:
        if pattern is None:
            return {
                "dataset": [0],
                "metric": [2, 3],
                "id": [4],
                "approach": [1]
            }
        split_pattern = pattern.split("_")
        return {
            key: [idx for idx, value in enumerate(
                split_pattern) if value == key]
            for key in np.unique(np.array(split_pattern))
        }

    def locate_file_position(reader_dict: dict, file: str) -> dict:
        split_file = file.split("_")
        return {
            key: "_".join([value for idx, value in enumerate(
                split_file) if idx in idx_list])
            for key, idx_list in reader_dict.items()
        }

    reader_dict = get_dict_from_pattern()
    results = {}
    for complete_file in os.listdir(args["input_path"]):
        # If it starts with test, skip
        if complete_file.startswith("test"):
            continue
        if complete_file.endswith(".csv"):
            file = complete_file.split("/")[-1].split(".")[0]
            current_reader = locate_file_position(reader_dict, file)

            # Level 0: dataset
            if current_reader["dataset"] not in results:
                results[current_reader["dataset"]] = {}
            # Level 1: (fairness) metric
            if current_reader["metric"] not in results[current_reader["dataset"]]:
                results[current_reader["dataset"]][current_reader["metric"]] = {}
            # Level 2: (sensitive attribute) id
            if current_reader["id"] not in results[current_reader["dataset"]][current_reader["metric"]]:
                results[current_reader["dataset"]][current_reader["metric"]
                                                   ][current_reader["id"]] = {}
            # Level 3: approach
            if current_reader["approach"] not in results[current_reader["dataset"]][current_reader["metric"]][current_reader["id"]]:
                results[current_reader["dataset"]][current_reader["metric"]
                                                   ][current_reader["id"]][current_reader["approach"]] = {}
            df = pd.read_csv(os.path.join(args["input_path"], complete_file))
            results[current_reader["dataset"]][current_reader["metric"]][current_reader["id"]][current_reader["approach"]][file.split(
                "_")[-1]] = df[sorted((list(df.columns)[-2:]))].to_numpy()

    return results, sorted(list(df.columns)[-2:])