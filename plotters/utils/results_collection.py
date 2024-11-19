import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_results(args: dict) -> (dict, list):
    def get_dict_from_pattern(pattern: str) -> dict:
        splitted_pattern = pattern.split("_")
        return {
            key: [idx for idx, value in enumerate(
                splitted_pattern) if value == key]
            for key in np.unique(np.array(splitted_pattern))
        }

    def locate_file_position(reader_dict: dict, file: str) -> dict:
        splitted_file = file.split("_")
        return {
            key: "_".join([value for idx, value in enumerate(
                splitted_file) if idx in idx_list])
            for key, idx_list in reader_dict.items()
        }

    reader_dict = get_dict_from_pattern(args["file_pattern"])
    results = {}
    for complete_file in os.listdir(args["input_path"]):
        if complete_file.endswith(".csv"):
            file = complete_file.split("/")[-1].split(".")[0]
            current_reader = locate_file_position(reader_dict, file)

            if current_reader["dataset"] not in results:
                results[current_reader["dataset"]] = {}
            if current_reader["approach"] not in results[current_reader["dataset"]]:
                results[current_reader["dataset"]
                        ][current_reader["approach"]] = {}
            if current_reader["metric"] not in results[current_reader["dataset"]][current_reader["approach"]]:
                results[current_reader["dataset"]][current_reader["approach"]
                                                   ][current_reader["metric"]] = {}

            df = pd.read_csv(os.path.join(args["input_path"], complete_file))
            results[current_reader["dataset"]][current_reader["approach"]][current_reader["metric"]][file.split(
                "_")[-1]] = df[sorted((list(df.columns)[-2:]))].to_numpy()

    return results, sorted(list(df.columns)[-2:])