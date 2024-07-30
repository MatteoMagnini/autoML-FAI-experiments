import sys
from datasets import create_cache_directory
from datasets.loader import load_dataset
from experiments.configurations import PATH as CONFIG_PATH, from_yaml_file_to_dict

if __name__ == '__main__':

    # Read configuration files
    conf_file_name = sys.argv[1]
    configuration_files = [CONFIG_PATH / conf_file_name]

    # sort the files
    configuration_files.sort()
    configurations = [from_yaml_file_to_dict(CONFIG_PATH / file_name) for file_name in configuration_files]
    create_cache_directory()

    for configuration in configurations:
        dataset = configuration["dataset"]
        train, test = load_dataset(dataset)
