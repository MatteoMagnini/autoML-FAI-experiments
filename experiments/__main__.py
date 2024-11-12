import sys
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from automl import plot_pareto, ResultSingleton
from automl.auto_cho import ChoMLP
from automl.auto_dpp import DPPMLP
from automl.auto_fauci import FauciMLP
from smac.multi_objective.parego import ParEGO
from automl.auto_jiang import JiangMLP
from automl.auto_prr import PRRMLP
from datasets import get_feature_data_type
from experiments.setup import PATH as CONFIG_PATH, from_yaml_file_to_dict, update_with_dataset
from results import save_incumbents

if __name__ == "__main__":
    conf_file_name = sys.argv[1]
    setup = from_yaml_file_to_dict(CONFIG_PATH / conf_file_name)
    setup = update_with_dataset(setup)
    setup["protected_type"] = get_feature_data_type(setup["dataset"], setup["protected"])
    setup["callbacks"] = []

    if setup["method"] == "fauci":
        mlp = FauciMLP(setup)
    elif setup["method"] == "jiang":
        mlp = JiangMLP(setup)
    elif setup["method"] == "cho":
        mlp = ChoMLP(setup)
    elif setup["method"] == "prr":
        mlp = PRRMLP(setup)
    elif setup["method"] == "dpp":
        mlp = DPPMLP(setup)
    else:
        raise ValueError(f"Unknown method {setup['method']}")
    objectives = ["1 - accuracy", "demographic_parity"]

    # Check if results are already available
    if ResultSingleton().check_if_results_exist(mlp.get_name()):
        print("Results already exist. Skipping...")
        #sys.exit(0)

    # Define our environment variables
    scenario = Scenario(
        mlp.configspace,
        objectives=objectives,
        walltime_limit=12*60*60,  # After 12 hour, we stop the hyperparameter optimization
        n_trials=10000,  # Evaluate max 10^4 different trials
        n_workers=1  # multiprocessing.cpu_count()
    )

    print(f"Using {scenario.n_workers} workers")

    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()

    # Get cost of default configuration
    # default_cost = smac.validate(mlp.configspace.get_default_configuration())
    # print(f"Validated costs from default config: \n--- {default_cost}\n")

    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        print("---", cost)
    save_incumbents(smac, incumbents, mlp.get_name() + "_incumbents.csv")
    ResultSingleton().save_results(mlp.get_name())

    # Let's plot a pareto front
    plot_pareto(smac, incumbents)
