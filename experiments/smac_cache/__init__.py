import json
from pathlib import Path

PATH = Path(__file__).parents[0]


def read_smac_cache_single_run(directory: Path) -> dict:
    """
    Reads a single SMAC cache file and returns the data as a dictionary.

    :param directory: The path to the SMAC cache file.
    :return: A dictionary containing the data from the SMAC cache file.
    """
    run_data = {}
    # go inside the only son directory (random hash name)
    # and again into the only run directory ("0"),
    # then read from runhistory.json
    son_directory = next(directory.iterdir())
    run_directory = next(son_directory.iterdir())
    runhistory_file = run_directory / "runhistory.json"
    if runhistory_file.exists():
        with runhistory_file.open("r") as file:
            run_data = file.read()
    else:
        print(f"Warning: {runhistory_file} does not exist.")
    # If the file is JSON, parse it
    try:
        run_data = json.loads(run_data)
    except json.JSONDecodeError:
        print(f"Error: {runhistory_file} is not a valid JSON file.")
        run_data = {}
    # Convert the run data to a dictionary
    if isinstance(run_data, str):
        run_data = {"error": "Invalid JSON format"}
    elif isinstance(run_data, dict):
        # Ensure all keys are strings
        run_data = {str(k): v for k, v in run_data.items()}
    else:
        run_data = {"error": "Unexpected data format"}
    # Ensure the run data is a dictionary
    if not isinstance(run_data, dict):
        print(f"Warning: {runhistory_file} does not contain a valid dictionary.")
        run_data = {"error": "Invalid run data format"}
    # Return the run data
    return run_data


def read_smac_cache_protected_runs(directory: Path) -> dict:
    """
    Reads all SMAC cache files in a directory and returns a dictionary of runs.

    :param directory: The path to the directory containing SMAC cache files.
    :return: A dictionary where keys are filenames and values are the parsed data.
    """
    runs = {}
    for sub_directory in directory.iterdir():
        if sub_directory.is_dir() and sub_directory.name != "__pycache__":
            runs[sub_directory.name] = read_smac_cache_single_run(sub_directory)
    # Return the dictionary of runs
    return runs


def read_smac_cache_fairness_runs(directory: Path) -> dict:
    """
    Reads all SMAC cache files in a directory and returns a dictionary of runs.

    :param directory: The path to the directory containing SMAC cache files.
    :return: A dictionary where keys are filenames and values are the parsed data.
    """
    runs = {}
    for sub_directory in directory.iterdir():
        if sub_directory.is_dir() and sub_directory.name != "__pycache__":
            run_data = read_smac_cache_protected_runs(sub_directory)
            # Use the sub-directory name as the key
            runs[sub_directory.name] = run_data
    # Return the dictionary of runs
    return runs


def read_smac_cache_dataset_runs(directory: Path) -> dict:
    """
    Reads all SMAC cache files in a directory and returns a dictionary of runs.

    :param directory: The path to the directory containing SMAC cache files.
    :return: A dictionary where keys are dataset names and values are dictionaries of runs.
    """
    runs = {}
    for sub_directory in directory.iterdir():
        if sub_directory.is_dir() and sub_directory.name != "__pycache__":
            run_data = read_smac_cache_fairness_runs(sub_directory)
            # Use the sub-directory name as the key
            runs[sub_directory.name] = run_data
    # Return the dictionary of runs
    return runs


def read_smac_cache_all_runs(directory: Path) -> dict:
    """
    Reads all SMAC cache files in a directory and returns a dictionary of runs.

    :param directory: The path to the directory containing SMAC cache files.
    :return: A dictionary where keys are dataset names and values are dictionaries of runs.
    """
    runs = {}
    for sub_directory in directory.iterdir():
        if sub_directory.is_dir() and sub_directory.name != "__pycache__":
            run_data = read_smac_cache_dataset_runs(sub_directory)
            # Use the sub-directory name as the key
            runs[sub_directory.name] = run_data
    # Return the dictionary of runs
    return runs