import numpy as np


def get_weightings(objectives_normed, df):
    """
    Returns the weighting used for the weighted importance. It uses the points on the pareto-front as weightings
    :param objectives_normed: the normalized objective names as a list of strings
    :param df: dataframe containing the encoded data
    :return: the weightings as a list of lists
    """
    optimized = is_pareto_efficient(df[objectives_normed].to_numpy())
    return df[optimized][objectives_normed].T.apply(lambda values: values / values.sum()).T.to_numpy()


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
    return is_efficient