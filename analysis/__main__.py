from results import PATH as RESULT_PATH
from analysis import PATH as ANALYSIS_PATH
import pandas as pd
import fire

def main():
    # 1. Read from [dataset]_[method]_[metric]_[metric]_[index]_incumbents.csv files in results folder
    # 2. For each dataset x method, compute the average values of the hyperparameters
    # 3. Save the results in [dataset]_best_hyperparameters.csv in analysis folder (the file is a matrix Hyperparameter x Methods)
    hyperparameters = ["batch_size", "lambda_value", "learning_rate", "number_of_layers", "number_of_neurons_per_layer"]
    results = {}
    for incumbent_file in RESULT_PATH.glob("*_incumbents.csv"):
        dataset, method, _, _, _, _ = incumbent_file.stem.split("_")
        if dataset not in results:
            results[dataset] = {}
        if method not in results[dataset]:
            results[dataset][method] = {}
        data = pd.read_csv(incumbent_file)
        for hyperparameter in hyperparameters:
            results[dataset][method][hyperparameter] = data[hyperparameter].mean()
    for dataset, dataset_dict in results.items():
        results[dataset] = pd.DataFrame(dataset_dict)
    for dataset, data in results.items():
        data.to_csv(ANALYSIS_PATH / f"{dataset}_best_hyperparameters.csv")


if __name__ == "__main__":
    fire.Fire(main)