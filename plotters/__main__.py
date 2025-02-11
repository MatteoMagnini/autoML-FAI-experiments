import os
from utils.pareto_front import plot_pareto_raw, plot_multiple_pareto_fronts
from utils.results_collection import read_results
from results import PATH as results_path


if __name__ == "__main__":
    args = {
        "input_path": results_path,
        "output_path": results_path,
        "file_pattern": "dataset_approach_metric_metric"
    }

    results, objectives = read_results(args)
    # print(results)

    for dataset, dataset_dict in results.items():
        print(dataset)
        pareto_fronts = {}
        for approach, approach_dict in dataset_dict.items():
            print(f"\t{approach}")
            for metric, metric_dict in approach_dict.items():
                print(f"\t\t{metric}")
                pareto_fronts[approach] = metric_dict["incumbents"]
                base_path = os.path.join(args["output_path"], f"{dataset}_{approach}_{metric}")
                plot_pareto_raw(costs=metric_dict["results"], pareto_costs=metric_dict["incumbents"],
                                file_paths=[base_path + ".eps", base_path + ".png"], obj0=objectives[0], obj1=objectives[1])
        base_path = os.path.join(args["output_path"], dataset)
        plot_multiple_pareto_fronts(
            pareto_fronts,
            title=f"Pareto Fronts for {dataset.capitalize()} dataset",
            obj0=objectives[0],
            obj1=objectives[1],
            file_paths=[base_path + ".eps", base_path + ".png"]
        )