from pathlib import Path
import yaml
from datasets.loader import load_dataset


PATH = Path(__file__).parents[0]
SEED = 0
K = 5
EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
ADULT_PATIENCE = 10
COMPAS_PATIENCE = 10
VERBOSE = 0


def from_yaml_file_to_dict(setup_file: str) -> dict:
    """
    Read the configuration file and return a dictionary.

    @param setup_file: configuration file
    @return: dictionary
    """
    with open(setup_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def update_with_dataset(setup: dict) -> dict:
    """
    Update the setup dictionary with the actual dataset.

    @param setup: setup dictionary
    @return: updated setup dictionary
    """
    dataset_name = setup["dataset"]
    train, test = load_dataset(dataset_name)
    setup["train"] = train
    setup["test"] = test
    return setup
