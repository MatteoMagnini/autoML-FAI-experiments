from pathlib import Path
import numpy as np
import yaml


PATH = Path(__file__).parents[0]
SEED = 0
K = 5
EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
ADULT_PATIENCE = 10
COMPAS_PATIENCE = 10
VERBOSE = 0


def from_yaml_file_to_dict(config_file: str) -> dict:
    """
    Read the configuration file and return a dictionary.

    @param config_file: configuration file
    @return: dictionary
    """
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_lambda_values(lambda_info: list[dict[str, float]]) -> list[list[float]]:
    lambda_info = {k: v for d in lambda_info for k, v in d.items()}
    max_lambda_values = lambda_info["max"]
    min_lambda_values = lambda_info["min"]
    step_lambda_values = lambda_info["step"]
    number_of_lists = len(max_lambda_values)
    lambda_lists_of_values = [
        [round(value, 5) for value in np.arange(min_lambda_values[i], max_lambda_values[i], step_lambda_values[i])] + [
            max_lambda_values[i]] for i in range(number_of_lists)]
    return lambda_lists_of_values
