import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from plotters.utils import PRETTY_NAMES
from sklearn.preprocessing import MinMaxScaler
from utils.pareto_front import plot_multiple_pareto_fronts, find_pareto_front
from utils.results_collection import read_results
from results import PATH as RESULTS_PATH
from plotters.test import PATH as PLOT_TEST_PATH
from plotters.parallel import PATH as PLOT_PARALLEL_PATH
from plotters.pareto import PATH as PLOT_PARETO_PATH
from tables import PATH as TABLES_PATH


COORDINATES = sorted([
    "batch_size",
    "lambda_value",
    "learning_rate",
    "number_of_layers",
    "number_of_neurons_per_layer",
])


def plot_pareto_fronts(results, compute_auc: bool = True):
    # Pareto fronts
    auc = []
    for dataset, dataset_dict in results.items():
        print(dataset)
        for metric, metric_dict in dataset_dict.items():
            print(f"\t\t{metric}")
            for attribute, attribute_dict in metric_dict.items():
                print(f"\t\t\t{attribute}")
                pareto_fronts_valid = {}
                for approach, approach_dict in attribute_dict.items():
                    print(f"\t\t\t\t{approach}")
                    pareto_fronts_valid[approach] = find_pareto_front(approach_dict["results"], '1 - accuracy', metric)

                    # Compute and store AUC if required
                    if compute_auc:
                        # Compute the AUC from the individual results
                        pareto_fronts_valid[approach]
                        auc_value = np.trapz(
                            y=pareto_fronts_valid[approach][metric].values,  # y-coordinates
                            x=pareto_fronts_valid[approach]['1 - accuracy'].values  # x-coordinates
                        )
                        if auc_value is not None:
                            auc.append({
                                "dataset": dataset,
                                "metric": metric,
                                "attribute": attribute,
                                "approach": approach,
                                "auc": auc_value
                            })

                base_path = os.path.join(PLOT_PARETO_PATH, f"{dataset}_{metric}_{attribute}")
                plot_multiple_pareto_fronts(
                    pareto_fronts_valid,
                    None,
                    title=f"Pareto Fronts for {dataset.capitalize()} dataset",
                    obj0=objectives[0],
                    obj1=metric,
                    file_paths=[base_path + ".eps", base_path + ".png"]
                )

    # Save AUC metrics to a CSV file
    if compute_auc and auc:
        auc_df = pd.DataFrame(auc)
        auc_csv_path = os.path.join(TABLES_PATH, "auc_metrics.csv")
        auc_df.to_csv(auc_csv_path, index=False)
        print(f"AUC metrics saved to {auc_csv_path}")


def plot_parallel_coordinates():
    # Parallel coordinates
    for file in os.listdir(RESULTS_PATH):
        if file.endswith("_results.csv") and not file.startswith("test"):
            dataset, approach, metric_left, metric_right, id, _ = file.split("_")
            metric = f"{metric_left}_{metric_right}"
            results = pd.read_csv(os.path.join(RESULTS_PATH, file))

            for cost in ["1 - accuracy", metric]:
                selected_cols = [cost] + COORDINATES
                df_selected = results.copy()[selected_cols].dropna()

                # Rinominare le colonne per una visualizzazione più chiara
                df_selected = df_selected.rename(columns=PRETTY_NAMES)
                df_selected_original = df_selected.copy()

                # Normalizzare ogni colonna, tranne la colonna della classe
                scaler = MinMaxScaler()
                for col in COORDINATES:
                    df_selected[PRETTY_NAMES[col]] = scaler.fit_transform(df_selected[[PRETTY_NAMES[col]]])

                # Creazione della figura
                plt.figure(figsize=(12, 6))
                sns.set_style("whitegrid")

                # Plottare direttamente con scale normalizzate
                parallel_coordinates(df_selected, class_column=PRETTY_NAMES[cost], colormap=plt.get_cmap("viridis"),
                                     alpha=0.7)

                # Ruota le etichette dell'asse X per renderle leggibili
                plt.xticks(rotation=45, ha="right")

                # Titolo del grafico
                plt.title(f"Parallel Coordinate Plot for\n{PRETTY_NAMES[approach]} on {dataset.capitalize()} dataset")

                # Nascondere la legenda
                plt.legend([], [], frameon=False)

                # Remove the y scale
                plt.yticks([])

                # Add the ORIGINAL scale for each y column
                # 5 ticks for each column
                for i, col in enumerate(COORDINATES):
                    min_val = df_selected_original[PRETTY_NAMES[col]].min()
                    max_val = df_selected_original[PRETTY_NAMES[col]].max()
                    for j in range(5):
                        t = plt.text(i, j / 4, f"{min_val + j * (max_val - min_val) / 4:.2f}", color="black",
                                     fontsize=8, ha="center", va="center")
                        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))

                # Aggiungere una barra dei colori sulla destra
                plt.subplots_adjust(right=0.8)
                cbar_ax = plt.gcf().add_axes([0.85, 0.2, 0.05, 0.6])
                cmap = plt.get_cmap("viridis")
                norm = plt.Normalize(df_selected[PRETTY_NAMES[cost]].min(), df_selected[PRETTY_NAMES[cost]].max())
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, cax=cbar_ax)
                cbar_ax.set_xlabel(PRETTY_NAMES[cost], rotation=0, labelpad=20, fontsize=10)

                # Salvare il grafico
                plot_path = os.path.join(PLOT_PARALLEL_PATH, f"{dataset}_{metric}_{id}_{approach}")
                plt.savefig(plot_path + ".eps", bbox_inches="tight")
                plt.savefig(plot_path + ".png", bbox_inches="tight", dpi=300)
                plt.close()

                # Clean up plt
                plt.clf()
                del df_selected, df_selected_original, scaler


def plot_pareto_fronts_test():
    # Pareto Test
    # Generate the pareto fronts like before, but this time for the test results only
    # You need to reed the configurations from the incumbents and then select those configurations from the test results
    # Then you can plot the pareto fronts
    pareto_fronts_test = {}
    pareto_fronts_valid = {}
    for file in os.listdir(RESULTS_PATH):
        if file.startswith("test"):
            _, dataset, approach, metric_left, metric_right, id, type = file.split("_")
            metric = f"{metric_left}_{metric_right}"
            validation_file = f"{dataset}_{approach}_{metric_left}_{metric_right}_{id}_incumbents.csv"
            validation_results = pd.read_csv(os.path.join(RESULTS_PATH, validation_file))
            test_results = pd.read_csv(os.path.join(RESULTS_PATH, file))
            # The test incumbents must have the same configurations as the validation incumbents (i.e., COORDINATES)
            configurations = validation_results[COORDINATES].to_dict(orient="records")
            test_results = find_pareto_front(test_results, "1 - accuracy", metric)
            test_results = test_results.dropna()
            test_results = test_results.sort_values("1 - accuracy", ascending=False)
            if dataset not in pareto_fronts_test:
                pareto_fronts_test[dataset] = {}
            if metric not in pareto_fronts_test[dataset]:
                pareto_fronts_test[dataset][metric] = {}
            if id not in pareto_fronts_test[dataset][metric]:
                pareto_fronts_test[dataset][metric][id] = {}
            if dataset not in pareto_fronts_valid:
                pareto_fronts_valid[dataset] = {}
            if metric not in pareto_fronts_valid[dataset]:
                pareto_fronts_valid[dataset][metric] = {}
            if id not in pareto_fronts_valid[dataset][metric]:
                pareto_fronts_valid[dataset][metric][id] = {}
            pareto_fronts_test[dataset][metric][id][approach] = test_results[["1 - accuracy", metric]].to_numpy()
            pareto_fronts_valid[dataset][metric][id][approach] = validation_results[["1 - accuracy", metric]].to_numpy()
    for dataset, dataset_dict in pareto_fronts_test.items():
        for metric, metric_dict in dataset_dict.items():
            for id, id_dict in metric_dict.items():
                base_path = os.path.join(PLOT_TEST_PATH, f"{dataset}_{metric}_{id}")
                plot_multiple_pareto_fronts(
                    pareto_fronts_test[dataset][metric][id],
                    None,
                    title=f"Pareto Fronts for {PRETTY_NAMES[dataset]} dataset",
                    obj0="1 - accuracy",
                    obj1=metric,
                    file_paths=[base_path + ".eps", base_path + ".png"]
                )

                # Plot the validation and test pareto fronts on the same graph with a line connecting the same configurations
                # in the validation and test results
                plot_multiple_pareto_fronts(
                    pareto_fronts_valid[dataset][metric][id],
                    pareto_fronts_test[dataset][metric][id],
                    title=f"Pareto Fronts for {PRETTY_NAMES[dataset]} dataset",
                    obj0="1 - accuracy",
                    obj1=metric,
                    file_paths=[base_path + "_validation_test.eps", base_path + "_validation_test.png"]
                )



if __name__ == "__main__":
    args = {
        "input_path": RESULTS_PATH,
    }
    results, objectives = read_results(args)
    plot_pareto_fronts(results)
    # plot_parallel_coordinates()
    # plot_pareto_fronts_test()
