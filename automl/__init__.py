from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from smac.facade.abstract_facade import AbstractFacade
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer
)


PATH = Path(__file__).parents[0]


class MLP:

    def __init__(self, setup: dict):
        self.setup = setup

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # TODO: add
        # - learning rate
        # - optimizer
        # - number of layers
        # - number of neurons per layer
        batch_size = Integer("batch_size", (32, 1024), default=32)
        lambda_value = Float("lambda_value", (0, 1), default=0.5)

        cs.add([batch_size, lambda_value])
        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:
        raise NotImplementedError


def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
    plt.show()
