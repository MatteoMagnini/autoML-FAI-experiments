import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from plotters.utils import PRETTY_NAMES
from sklearn.preprocessing import MinMaxScaler
from utils.pareto_front import plot_pareto_raw, plot_multiple_pareto_fronts
from utils.results_collection import read_results
from results import PATH as RESULTS_PATH


COORDINATES = sorted([
    "batch_size",
    "lambda_value",
    "learning_rate",
    "number_of_layers",
    "number_of_neurons_per_layer",
])


if __name__ == "__main__":
    args = {
        "input_path": RESULTS_PATH,
        "output_path": RESULTS_PATH,
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

    # Iterate over the results files *_results.csv in the results folder
    for file in os.listdir(RESULTS_PATH):
        if file.endswith("_results.csv"):
            dataset, approach, metric_left, metric_right, id, _ = file.split("_")
            metric = f"{metric_left}_{metric_right}"
            results = pd.read_csv(os.path.join(RESULTS_PATH, file))

            for cost in ["1 - accuracy", "demographic_parity"]:
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
                plot_path = os.path.join(RESULTS_PATH, f"{dataset}_{approach}_{metric}_{cost}_parallel_coordinates")
                plt.savefig(plot_path + ".eps", bbox_inches="tight")
                plt.savefig(plot_path + ".png", bbox_inches="tight", dpi=300)
                plt.close()

                # Clean up plt
                plt.clf()
                del df_selected, df_selected_original, scaler