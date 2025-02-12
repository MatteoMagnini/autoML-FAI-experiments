from deepcave.runs.converters.smac3v2 import SMAC3v2Run
from deepcave.runs.group import Group
import os


def read_runs(path):
    """
    Read runs from a given path and returns them as a group.
    :param path: path to the smac runs
    :return: the group of smac runs
    """
    runs = list()
    for folder in os.listdir(path):
        subpath = os.path.join(path, folder)
        try:
            print('Reading', folder)
            runs.append(SMAC3v2Run.from_path(subpath))
        except Exception:
            for subfolder in os.listdir(subpath):
                try:
                    print('Reading', subfolder)
                    runs.append(SMAC3v2Run.from_path(os.path.join(subpath, subfolder)))
                except Exception as e:
                    print(subfolder, 'could not be read.')
                    print(e)
                    continue
    if len(runs)==0:
        print('Data could not be read.')
        exit()
    return Group('group', runs)


def normalize(values, obj):
    """
    Normalize the input values with min-max normalization.
    :param values: the values to be normalized
    :param obj: the corresponding objective id
    :return: the normalized values
    """
    global norm_dict
    norm_dict[obj] = (values.min(), values.max())
    return (values - norm_dict[obj][0]) / (norm_dict[obj][1] - norm_dict[obj][0])


def normalize_ablation(values, obj):
    """
    Normalize the input values by the same normalization of the input objective id.
    :param values: the values to be normalized
    :param obj: the corresponding objective id
    :return: the normalized values
    """
    global norm_dict
    normed = (values - norm_dict[obj][0]) / (norm_dict[obj][1] - norm_dict[obj][0])
    normed[normed < 0] = 0
    return normed


def encode_and_normalize(group, objectives):
    """
    Encodes the configurations and normalizes the objectives.
    :param group: the runs as group
    :param objectives: the objectives to weight as list
    :return: the dataframe containing the encoded data
    """
    df = group.get_encoded_data(objectives, budget=None, specific=True, include_combined_cost=False,
                                include_config_ids=True)
    df = df.dropna(subset=[o.name for o in objectives])
    obj = 0
    global norm_dict
    norm_dict = dict()
    for o in objectives:
        df[o.name + '_normed'] = normalize(df[o.name], obj)
        obj += 1
    return df