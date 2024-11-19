import os

from utils.pareto_front import plot_pareto_raw
from utils.results_collection import read_results

if __name__ == "__main__":
    args = {
        "input_path": "results",
        "output_path": "results",
        "file_pattern": "dataset_approach_metric_metric"
    }

    results, objectives = read_results(args)
    # print(results)

    for dataset, dataset_dict in results.items():
        print(dataset)
        for approach, approach_dict in dataset_dict.items():
            print(f"\t{approach}")
            for metric, metric_dict in approach_dict.items():
                print(f"\t\t{metric}")
                plot_pareto_raw(costs=metric_dict["results"], pareto_costs=metric_dict["incumbents"],
                                file_path=os.path.join(args["output_path"], f"{dataset}_{approach}_{metric}.png"), obj0=objectives[0], obj1=objectives[1])