import pandas as pd
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators.fanova import fANOVA
from deepcave.runs import AbstractRun


class fANOVAWeighted(fANOVA):
    """
    Calculate and provide midpoints and sizes from the forest's split values in order to get the marginals.
    Overriden to train the random forest with an arbitrary weighting of the objectives.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        super().__init__(run)
        self.n_trees = 100

    def train_model(
            self,
            group, df, objectives_normed, weighting
    ) -> None:
        """
        Train a FANOVA Forest model where the objectives are weighted by the input weighting.
        :param group: the runs as group
        :param df: dataframe containing the encoded data
        :param objectives_normed: the normalized objective names as a list of strings
        :param weighting: the weighting as list
        """
        X = df[group.configspace.get_hyperparameter_names()].to_numpy()
        Y = sum(df[obj] * weighting for obj, weighting in zip(objectives_normed, weighting)).to_numpy()

        self._model = FanovaForest(self.cs, n_trees=self.n_trees, seed=0)
        self._model.train(X, Y)


def do_weighted_fanova_mo(group, df, objectives_normed, weightings):
    """
    Calculates weighted fAnova for multiple objectives.
    :param group: the runs as group
    :param df: dataframe containing the encoded data
    :param objectives_normed: the normalized objective names as a list of strings
    :param weightings: the weightings as list of lists
    :return: the dataframe containing the importances for each hyperparameter per weighting
    """
    df_all = pd.DataFrame([])
    for w in weightings:
        result = fANOVAWeighted(group)
        result.train_model(group, df, objectives_normed, w)
        df_res = pd.DataFrame(result.get_importances(hp_names=None)).loc[0:1].T.reset_index()
        df_res['weight_for_' + objectives_normed[0]] = w[0]
        df_all = pd.concat([df_all, df_res])
    df_all = df_all.rename(columns={0: 'fanova', 1:'variance', 'index': 'hp_name'})
    return df_all