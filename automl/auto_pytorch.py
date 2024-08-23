import time
import warnings
from ConfigSpace import Configuration
from sklearn.model_selection import KFold
from _logging import logger
from automl import MLP
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset
from experiments import PytorchNN, PyTorchConditions, evaluate_predictions


class PytorchMLP(MLP):

    features_to_drop = 1

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, n_epochs, batch_size, conditions):
        raise NotImplementedError("Method not implemented")

    def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:
        train = self.setup["train"]
        test = self.setup["test"]
        epochs = self.setup["epochs"]
        protected = self.setup["protected"]
        results = {}

        folds = KFold(n_splits=5, shuffle=True, random_state=seed)
        for exp_number, (train_idx, valid_idx) in enumerate(folds.split(train)):
            train_data, valid_data = train.iloc[train_idx], train.iloc[valid_idx]
            pt_dataset = FairnessPyTorchDataset(train_data, valid_data, test)
            pt_dataset.prepare_ndarray(protected)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = PytorchNN(n_inputs=train_data.shape[1] - self.features_to_drop)
                callbacks = PyTorchConditions(model, epochs)

                start_time = time.time()
                predictions = self.train_and_predict_classifier(
                    dataset=pt_dataset,
                    net=model,
                    metric=self.setup["fairness_metric"],
                    lambda_=config.get("lambda_value"),
                    n_epochs=epochs,
                    batch_size=config.get("batch_size"),
                    conditions=callbacks
                )
                end_time = time.time()
                logger.debug(f"end training model")
                logger.debug(f"training time: {end_time - start_time}")
                test_y = test.iloc[:, -1].reset_index(drop=True)
                test_p = test.iloc[:, protected].reset_index(drop=True)
                tmp_results = evaluate_predictions(test_p, predictions, test_y, logger)
                # tmp_results["training_time"] = end_time - start_time
                results = {k: results.get(k, 0) + tmp_results.get(k, 0) for k in set(results) | set(tmp_results)}
            results = {k: v / 5 for k, v in results.items()}
        return results
