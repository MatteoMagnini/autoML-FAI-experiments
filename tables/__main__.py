import pandas as pd
from tables import PATH as TABLES_PATH
from plotters.utils import PRETTY_NAMES


def custom_latex_table(table: pd.DataFrame, caption: str, label: str) -> str:
    """
    Convert a pivoted DataFrame to LaTeX with custom multicolumn headers.
    """
    import io

    buf = io.StringIO()
    buf.write("\\begin{table}[ht]\n\\centering\n")
    buf.write("\\begin{tabular}{l|ccc|ccc}\n")
    buf.write("\\toprule\n")
    buf.write("\\textbf{Approach} & \\multicolumn{3}{c|}{\\textbf{Adult}} & \\multicolumn{3}{c}{\\textbf{COMPAS}} \\\\\n")
    buf.write(" & Sex & Ethnicity & Intersec. & Sex & Ethnicity & Intersec. \\\\\n")
    buf.write("\\midrule\n")

    for approach, row in table.iterrows():
        row_str = [approach] + [f"{x:.4f}" if pd.notna(x) else "-" for x in row]
        buf.write(" & ".join(row_str) + " \\\\\n")

    buf.write("\\bottomrule\n")
    buf.write("\\end{tabular}\n")
    buf.write(f"\\caption{{{caption}}}\n")
    buf.write(f"\\label{{{label}}}\n")
    buf.write("\\end{table}\n")

    return buf.getvalue()


def generate_latex_auc_table(name: str = "auc_metrics.csv") -> None:
    """
    Generates a LaTeX table for AUC results, grouped by dataset and attribute.
    """

    # Read the CSV file
    df = pd.read_csv(TABLES_PATH / name)

    # Mapping of attribute codes to names
    protected_index_to_feature_name = {
        "8": "Sex",
        "7": "Ethnicity",
        "8+7": "Intersec.",
        "1": "Sex",
        "5": "Ethnicity",
        "1+5": "Intersec.",
    }

    # Dataset pretty names
    dataset_pretty_names = {
        "adult": "Adult",
        "compas": "COMPAS",
    }

    # Define desired order
    ordered_datasets = ["adult", "compas"]
    ordered_attributes = ["8", "7", "8+7", "1", "5", "1+5"]

    # Get unique metrics
    metrics = df["metric"].unique()

    for metric in metrics:
        metric_df = df[df["metric"] == metric]

        # Pivot
        table = metric_df.pivot_table(
            index="approach",
            columns=["dataset", "attribute"],
            values="auc",
            aggfunc="first",
        )

        # Pretty approach names
        table.index = [PRETTY_NAMES.get(a, a) for a in table.index]

        # Reorder columns
        new_cols = []
        for d in ordered_datasets:
            for a in ordered_attributes:
                if (d, a) in table.columns:
                    new_cols.append((d, a))
        table = table[new_cols]

        # Pretty column names
        table.columns = [
            (dataset_pretty_names[d], protected_index_to_feature_name.get(a, a))
            for d, a in table.columns
        ]

        latex_str = custom_latex_table(table,
                                       caption=f"Area under the curve (AUC) for {PRETTY_NAMES[metric]} on the Adult and COMPAS datasets.\n"
                                                "%\n"
                                                "The best results are highlighted in bold.\n",
                                       label=f"tab:{metric}-auc")
        with open(TABLES_PATH / f"{metric}_auc_table.tex", "w") as f:
            f.write(latex_str)


if __name__ == "__main__":
    generate_latex_auc_table()