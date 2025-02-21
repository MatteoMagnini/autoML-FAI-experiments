import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# SMAC stuff
from ConfigSpace import Configuration
from smac.facade.abstract_facade import AbstractFacade
from plotters.utils import PRETTY_NAMES, COLOR_MAP


def find_pareto_front(results: pd.DataFrame or np.ndarray, obj0: str, obj1: str) -> pd.DataFrame:
    """Finds the Pareto front of the results DataFrame.

    Parameters
    ----------
    results : pd.DataFrame
        The DataFrame containing the results.
    obj0 : str
        The name of the first objective.
    obj1 : str
        The name of the second objective.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the Pareto front.
    """
    if isinstance(results, np.ndarray):
        results = pd.DataFrame(results, columns=[obj0, obj1])

        # Sort: First objective ascending, second objective ascending (per facilitare il controllo dei dominati)
    results = results.sort_values(by=[obj0, obj1], ascending=[True, True], ignore_index=True)

    pareto_front = []
    best_so_far = float("inf")  # Best (minimum) value of obj1 found so far
    last_obj0 = None  # Per tenere traccia dell'ultimo valore di obj0 inserito nel fronte

    # Sweep attraverso i punti ordinati
    for _, row in results.iterrows():
        if row[obj0] != last_obj0 and row[obj1] < best_so_far:
            pareto_front.append(row)
            best_so_far = row[obj1]
            last_obj0 = row[obj0]

    pareto_front = pd.DataFrame(pareto_front)

    # Drop duplicates (non dovrebbe servire con la logica attuale, ma per sicurezza)
    pareto_front = pareto_front.drop_duplicates(subset=[obj0, obj1])

    # Rimuove i corner cases (dove uno dei due obiettivi è 0)
    pareto_front = pareto_front[(pareto_front[obj0] != 0) & (pareto_front[obj1] != 0)]

    return pareto_front


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
        file_paths=file_path,
        obj0=smac.scenario.objectives[0],
        obj1=smac.scenario.objectives[0]
    )


def plot_pareto_raw(
    costs: dict,
    pareto_costs: dict,
    file_paths: os.path or list[os.path],
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
        file_paths=file_paths,
        obj0=obj0,
        obj1=obj1
    )  
    

def plot_pareto(
    costs_x: dict,
    costs_y: dict,
    pareto_costs_x: dict,
    pareto_costs_y: dict,
    file_paths: os.path or list[os.path],
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

    ax.set_xlabel(PRETTY_NAMES[obj0], fontsize=16)
    ax.set_ylabel(PRETTY_NAMES[obj1], fontsize=16)
    # plt.show()
    if isinstance(file_paths, list):
        for file_path in file_paths:
            plt.savefig(file_path)
    else:
        plt.savefig(file_paths)


def plot_multiple_pareto_fronts(
        methods_incumbents: dict[str, dict],
        methods_incumbents_test: None or dict[str, dict],
        title: str,
        obj0: str,
        obj1: str,
        file_paths: os.path or list[os.path],
) -> None:
    """
    Plots the Pareto frontiers for multiple methods on the same graph, with points and enhanced visual clarity.

    Parameters
    ----------
    methods_incumbents : dict[str, dict]
        A dictionary with method names as keys and dictionaries of incumbents as values.
        The incumbents are dictionaries with objective names as keys and lists of incumbents as values.
    methods_incumbents_test : None or dict[str, dict]
        A dictionary with method names as keys and dictionaries of test incumbents as values.
        The test incumbents are dictionaries with objective names as keys and lists of test incumbents as values.
    title : str
        The title of the plot.
    obj0 : str
        The name of the first objective.
    obj1 : str
        The name of the second objective.
    file_paths : os.path or list[os.path]
        The path where to save the plot
    """
    plt.figure(figsize=(10, 6))


    for idx, (method_name, incumbents) in enumerate(methods_incumbents.items()):
        costs = np.array(incumbents)

        # Check if costs are 2D
        if costs.shape[1] != 2:
            raise ValueError(f"Expected 2D costs, but got {costs.shape[1]}D data for method '{method_name}'.")

        # Sort costs based on 0th value, in case of ties, sort based on 1st value
        # The fist one MUST be ascending, the second one MUST be descending
        sorted_indices = np.lexsort((costs[:, 1], -costs[:, 0]))
        sorted_costs = costs[sorted_indices]

        # Plot the Pareto frontier line
        plt.plot(
            sorted_costs[:, 0],
            sorted_costs[:, 1],
            label=f"{PRETTY_NAMES[method_name]} Pareto frontier",
            color=COLOR_MAP[PRETTY_NAMES[method_name]],
            linewidth=2.5,
            linestyle='-'
        )
        # Plot the points of the frontier
        plt.scatter(
            sorted_costs[:, 0],
            sorted_costs[:, 1],
            color=COLOR_MAP[PRETTY_NAMES[method_name]],
            edgecolors='black',
            s=50,
            label=f"{PRETTY_NAMES[method_name]} incumbents"
        )

        # Plot the test points if available
        if methods_incumbents_test is not None:
            test_costs = np.array(methods_incumbents_test[method_name])
            sorted_indices = np.lexsort((test_costs[:, 1], -test_costs[:, 0]))
            test_costs = test_costs[sorted_indices]
            if test_costs.shape[1] != 2:
                raise ValueError(f"Expected 2D costs, but got {test_costs.shape[1]}D data for method '{method_name}'.")
            plt.plot(
                test_costs[:, 0],
                test_costs[:, 1],
                color=COLOR_MAP[PRETTY_NAMES[method_name]],
                label=f"{PRETTY_NAMES[method_name]} test Pareto frontier",
                linewidth=2.5,
                linestyle='--'
            )
            plt.scatter(
                test_costs[:, 0],
                test_costs[:, 1],
                color=COLOR_MAP[PRETTY_NAMES[method_name]],
                edgecolors='black',
                s=50,
                marker='x',
                label=f"{PRETTY_NAMES[method_name]} test incumbents"
            )

    # Plot settings
    # plt.title(title)
    plt.xlabel(PRETTY_NAMES[obj0], fontsize=14)
    plt.ylabel(PRETTY_NAMES[obj1], fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Methods", loc="best", fontsize='medium')
    plt.tight_layout()
    if isinstance(file_paths, list):
        for file_path in file_paths:
            plt.savefig(file_path)
    else:
        plt.savefig(file_paths)
    plt.close()
