import os

import matplotlib.pyplot as plt
import numpy as np


# SMAC stuff
from ConfigSpace import Configuration
from smac.facade.abstract_facade import AbstractFacade


PRETTY_NAMES = {
    "1 - accuracy": "1 - Accuracy",
    "demographic_parity": "Demographic Parity",
    "equal_odds": "Equal Odds",
    "disparate_impact": "Disparate Impact",
    "fauci": "FaUCI",
    "jiang": "GDP",
    "cho": "KDE",
    "prr": "PRR",
}


def get_pareto_front(smac: AbstractFacade) -> tuple[list[Configuration], list[list[float]]]:
    """Returns the Pareto front of the runhistory.

    Returns
    -------
    configs : list[Configuration]
        The configs of the Pareto front.
    costs : list[list[float]]
        The costs from the configs of the Pareto front.
    """

    # Get costs from runhistory first
    average_costs = []
    configs = smac.runhistory.get_configs()
    for config in configs:
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each
        # configuration
        # Luckily, SMAC already does this for us
        average_cost = smac.runhistory.average_cost(config)
        average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)

    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1

    return [configs[i] for i in is_efficient], [average_costs[i] for i in is_efficient]

def plot_pareto_smac(smac: AbstractFacade, file_path: os.path) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    # Get Pareto costs
    # print([smac.runhistory.get_cost(incumbent) for incumbent in incumbents])
    _, c = get_pareto_front(smac)
    pareto_costs = np.array(c)

    # Sort them a bit
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]

    # Get all other costs from runhistory
    average_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if average_cost not in c:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plot_pareto(
        costs_x=costs_x,
        costs_y=costs_y,
        pareto_costs_x=pareto_costs_x,
        pareto_costs_y=pareto_costs_y,
        file_path=file_path,
        obj0=smac.scenario.objectives[0],
        obj1=smac.scenario.objectives[0]
    )


def plot_pareto_raw(
    costs: dict,
    pareto_costs: dict,
    file_path: os.path,
    obj0: str,
    obj1: str
) -> None:

    # Let's work with a numpy array
    # costs = np.vstack(summary["costs"])
    # pareto_costs = np.vstack(summary["pareto_costs"])
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them
    
    # print(file_path)
    costs_x, costs_y = costs[:, 0], costs[:, 1]
    # print(costs)
    # print(costs_x)
    # print()
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    # print(pareto_costs)
    # print(pareto_costs_x)
    # print()

    plot_pareto(
        costs_x=costs_x,
        costs_y=costs_y,
        pareto_costs_x=pareto_costs_x,
        pareto_costs_y=pareto_costs_y,
        file_path=file_path,
        obj0=obj0,
        obj1=obj1
    )  
    

def plot_pareto(
    costs_x: dict,
    costs_y: dict,
    pareto_costs_x: dict,
    pareto_costs_y: dict,
    file_path: os.path,
    obj0: str,
    obj1: str
) -> None:

    fig, ax = plt.subplots()

    ax.scatter(costs_x, costs_y, marker="x")
    ax.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r")
    ax.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(pareto_costs_x)],  # We add bounds
        [np.max(pareto_costs_y)] + pareto_costs_y.tolist() + \
        [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )
    dataset, method = file_path.split("/")[-1].split(".")[0].split("_")[:2]
    method = PRETTY_NAMES[method]
    # ax.set_title("Results for " + method + " on " + dataset)
    ax.set_xlabel(PRETTY_NAMES[obj0], fontsize=16)
    ax.set_ylabel(PRETTY_NAMES[obj1], fontsize=16)
    # plt.show()
    fig.savefig(file_path)


def plot_multiple_pareto_fronts(
        methods_incumbents: dict[str, dict],
        title: str,
        obj0: str,
        obj1: str,
        file_path: os.path,
) -> None:
    """
    Plots the Pareto frontiers for multiple methods on the same graph, with points and enhanced visual clarity.

    Parameters
    ----------
    methods_incumbents : dict
        A dictionary where keys are method names and values are lists of incumbents (configurations and costs).
    title : str
        Title of the plot.
    obj0 : str
        Label for the X-axis.
    obj1 : str
        Label for the Y-axis.
    file_path : os.path
        Path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Enhanced colormap with accessible and visually appealing colors
    colors = plt.cm.get_cmap('Set2', len(methods_incumbents))

    for idx, (method_name, incumbents) in enumerate(methods_incumbents.items()):
        costs = np.array(incumbents)

        # Check if costs are 2D
        if costs.shape[1] != 2:
            raise ValueError(f"Expected 2D costs, but got {costs.shape[1]}D data for method '{method_name}'.")

        # Sort costs
        sorted_indices = np.argsort(costs[:, 0])
        sorted_costs = costs[sorted_indices]

        # Plot the Pareto frontier line
        plt.plot(
            sorted_costs[:, 0],
            sorted_costs[:, 1],
            label=PRETTY_NAMES[method_name],
            color=colors(idx),
            linewidth=2.5,
            linestyle='-'
        )
        # Plot the points of the frontier
        plt.scatter(
            sorted_costs[:, 0],
            sorted_costs[:, 1],
            color=colors(idx),
            edgecolors='black',
            s=50,
            label=f"{PRETTY_NAMES[method_name]} points"
        )

    # Plot settings
    # plt.title(title)
    plt.xlabel(PRETTY_NAMES[obj0], fontsize=14)
    plt.ylabel(PRETTY_NAMES[obj1], fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Methods", loc="best", fontsize='medium')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
