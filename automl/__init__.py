from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from smac import HyperparameterOptimizationFacade as HPOFacade
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer, Categorical
)
from results import PATH as RESULT_PATH


PATH = Path(__file__).parents[0]


def get_conf_space_info(config) -> dict[str, int | float]:
    return {
        "batch_size": config["batch_size"],
        "lambda_value": config["lambda_value"],
        "learning_rate": config["learning_rate"],
        "number_of_layers": config["number_of_layers"],
        "number_of_neurons_per_layer": config["number_of_neurons_per_layer"],
    }


class MLP:

    def __init__(self, setup: dict):
        self.setup: dict = setup

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        # The following line rises an error! Why? It says that cannot be serialized.
        # batch_size = Categorical("batch_size", [32, 64, 128, 256, 512, 1024], default=32)
        batch_size = Integer("batch_size", (32, 1024), default=32)
        lambda_value = Float("lambda_value", (0, 1), default=0.5)
        learning_rate = Float("learning_rate", (1e-4, 1e-1), default=1e-3, log=True)
        number_of_layers = Integer("number_of_layers", (1, 5), default=1)
        number_of_neurons_per_layer = Integer("number_of_neurons_per_layer", (64, 1024), default=64, log=True)

        cs.add([batch_size, lambda_value, learning_rate, number_of_layers, number_of_neurons_per_layer])
        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 50) -> dict[str, float]:
        raise NotImplementedError

    def get_name(self) -> str:
        return f"{self.setup['dataset']}_{self.setup['method']}_{self.setup['fairness_metric']}_{self.setup['protected']}"


def plot_pareto(smac: HPOFacade, incumbents: list[Configuration]) -> None:
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
    if len(average_costs) == 1:
        costs = np.array(average_costs)
    else:
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


class ResultSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultSingleton, cls).__new__(cls)
            cls._instance.results = []  # List of dictionaries with the results.

        return cls._instance

    def append(self, result: dict[int: dict[str: float]]) -> None:
        self.results.append(result)

    def save_results(self, name: str) -> None:
        # Convert to DataFrame
        # Every dictionary has the same keys
        df = pd.DataFrame(self.results)
        df.to_csv(RESULT_PATH / f"{name}_results.csv", index=False)

    def check_if_results_exist(self, name: str) -> bool:
        return (RESULT_PATH / f"{name}_results.csv").exists()
