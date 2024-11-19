import os

import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_raw(
    costs: dict,
    pareto_costs: dict,
    file_path: os.path,
    obj0: str,
    obj1: str) -> None:

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
    obj1: str) -> None:

    fig, ax = plt.subplots()

    ax.scatter(costs_x, costs_y, marker="x")
    ax.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r")
    ax.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(pareto_costs_x)
                                                         ],  # We add bounds
        [np.max(pareto_costs_y)] + pareto_costs_y.tolist() + \
        [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    ax.set_title(" ".join(file_path.split(
        "/")[-1].split(".")[0].split("_")).title())
    ax.set_xlabel(obj0)
    ax.set_ylabel(obj1)
    # plt.show()
    fig.savefig(file_path)

