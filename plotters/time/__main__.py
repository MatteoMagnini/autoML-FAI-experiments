from plotters.utils import PRETTY_NAMES, COLOR_MAP
from experiments.smac_cache import PATH as SMAC_PATH, read_smac_cache_all_runs
import matplotlib.pyplot as plt
import seaborn as sns


def rearrange_data(original_dict):
    rearranged_dict = {}
    for dataset_name, fairness_metrics in original_dict.items():
        for fairness_metric, sensitive_attributes in fairness_metrics.items():
            if fairness_metric not in rearranged_dict:
                rearranged_dict[fairness_metric] = {}
            if dataset_name not in rearranged_dict[fairness_metric]:
                rearranged_dict[fairness_metric][dataset_name] = {}
            for sensitive_attribute, approaches in sensitive_attributes.items():
                if sensitive_attribute not in rearranged_dict[fairness_metric][dataset_name]:
                    rearranged_dict[fairness_metric][dataset_name][sensitive_attribute] = {}
                rearranged_dict[fairness_metric][dataset_name][sensitive_attribute] = approaches
    return rearranged_dict


def generate_violin_plot(fairness: str, smac_cache_dataset_runs: dict):
    """
    Generates a figure with multiple violin plots for the specified fairness metric.
    In particular the figure is organised in a 3x2 grid (protected attribute x dataset).
    Each subplot contains a violin plot for the different approaches measuring the computational time.

    :param fairness: The fairness metric to plot.
    :param smac_cache_dataset_runs: The dataset runs from the SMAC cache.
    """

    # Set the style for the plots
    sns.set(style="whitegrid")

    # Create a figure with a 3x2 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()

    # Iterate over datasets and their sensitive attributes
    # sort: Adult first and then Compas
    title_settled = False
    for i, (dataset_name, sensitive_attributes) in enumerate(sorted(smac_cache_dataset_runs.items())):
        sorted_sensitive_attributes = ["Sex", "Ethnicity", "Intersectionality"]  # Explicitly sort sensitive attributes
        for j, (sensitive_attribute, approaches) in enumerate(sorted(sensitive_attributes.items(), key=lambda x: sorted_sensitive_attributes.index(PRETTY_NAMES[x[0]]))):
            # Prepare data for the violin plot
            data = []
            labels = []
            for approach_name, approach_data in approaches.items():
                data.append([x[5] for x in approach_data["data"]])
                labels.append(PRETTY_NAMES[approach_name])

            # Create the violin plot
            sns.violinplot(data=data, ax=axs[j * 2 + i], inner="quartile",
                           color=COLOR_MAP.get(PRETTY_NAMES[sensitive_attribute]), scale="count", cut=0)

            axs[j * 2].set_ylabel(PRETTY_NAMES[sensitive_attribute], labelpad=20, fontsize=20)
            # axs[j * 2 + i].set_ylabel("Time (seconds)")
            axs[j * 2 + i].set_xticklabels(labels, rotation=45)
        axs[4 + i].set_xlabel(f"{PRETTY_NAMES[dataset_name]}", fontsize=20, labelpad=20)


    # Adjust layout and show the plot
    plt.tight_layout()
    # Save the figure
    plt.savefig(f"violin_plot_{fairness}.eps")
    plt.savefig(f"violin_plot_{fairness}.png")




if __name__ == "__main__":
    all_smac_cache = read_smac_cache_all_runs(SMAC_PATH, True)
    all_smac_cache = rearrange_data(all_smac_cache)
    for metric in all_smac_cache.keys():
        generate_violin_plot(metric, all_smac_cache[metric], )
