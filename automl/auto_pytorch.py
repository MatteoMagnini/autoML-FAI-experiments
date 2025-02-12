import time
import warnings

from numpy.ma.setup import configuration

from experiments.cache import PATH as CACHE_PATH
import torch
from ConfigSpace import Configuration
from sklearn.model_selection import KFold
from _logging import logger
from automl import MLP, ValidResultSingleton, get_conf_space_info, TestResultSingleton
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset
from experiments import PytorchNN, PyTorchConditions, evaluate_predictions


class PytorchMLP(MLP):

    features_to_drop = 1

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions, on_test=False):
        raise NotImplementedError("Method not implemented")

    def train(self, config: Configuration, seed: int = 0, budget: int = 100) -> dict[str, float]:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        train = self.setup["train"]
        test = self.setup["test"]
        epochs = self.setup["epochs"]
        protected = self.setup["protected"]
        results = {}

        folds = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_results = []
        valid_singleton = ValidResultSingleton()
        test_singleton = TestResultSingleton()
        for exp_number, (train_idx, valid_idx) in enumerate(folds.split(train)):
            train_data, valid_data = train.iloc[train_idx], train.iloc[valid_idx]
            pt_dataset = FairnessPyTorchDataset(train_data, valid_data, test)
            pt_dataset.prepare_ndarray(protected)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = PytorchNN(
                    inputs=train_data.shape[1] - self.features_to_drop,
                    hidden_layers=config.get("number_of_layers"),
                    neurons=config.get("number_of_neurons_per_layer"),
                )
                model.to(device)
                callbacks = PyTorchConditions(model, epochs)

                start_time = time.time()
                predictions = self.train_and_predict_classifier(
                    net=model,
                    dataset=pt_dataset,
                    batch_size=config.get("batch_size"),
                    conditions=callbacks,
                    lambda_=config.get("lambda_value"),
                    lr=config.get("learning_rate"),
                    metric=self.setup["fairness_metric"],
                    n_epochs=epochs,
                    on_test=False,
                )
                # Save predictions into a txt file in the cache folder
                # Than place the file into the right subfolder, the name of the subfolder is the setup configuration
                # The name of the file is an incremental number
                sub_path = CACHE_PATH / self.get_sub_path()
                config_text = f"{config.get('batch_size')}_{config.get('lambda_value')}_{config.get('learning_rate')}_{config.get('number_of_layers')}_{config.get('number_of_neurons_per_layer')}"
                file_name = f"{config_text}_{exp_number}.txt"
                # Recursively create the subfolder if it does not exist
                sub_path.mkdir(parents=True, exist_ok=True)
                with open(sub_path / file_name, "w") as f:
                    f.write("\n".join(map(str, predictions)))
                end_time = time.time()
                logger.debug(f"end training model")
                logger.debug(f"training time: {end_time - start_time}")
                valid_y = valid_data.iloc[:, -1].reset_index(drop=True)
                valid_p = valid_data.iloc[:, protected].reset_index(drop=True)
                fold_results.append(evaluate_predictions(valid_p, predictions, valid_y, logger))

            fairness_metric = sum([result["demographic_parity"] for result in fold_results]) / 5
            one_minus_accuracy = sum([result["1 - accuracy"] for result in fold_results]) / 5
            results = {
                "demographic_parity": fairness_metric,
                "1 - accuracy": one_minus_accuracy
            }

        conf_info = get_conf_space_info(config)
        valid_singleton.append(conf_info | results)

        # Train on the entire training set a new model with the same configuration
        pt_dataset = FairnessPyTorchDataset(train, test, test)
        pt_dataset.prepare_ndarray(protected)
        model = PytorchNN(
            inputs=train.shape[1] - self.features_to_drop,
            hidden_layers=config.get("number_of_layers"),
            neurons=config.get("number_of_neurons_per_layer"),
        )
        model.to(device)
        callbacks = PyTorchConditions(model, epochs)

        start_time = time.time()
        predictions = self.train_and_predict_classifier(
            net=model,
            dataset=pt_dataset,
            batch_size=config.get("batch_size"),
            conditions=callbacks,
            lambda_=config.get("lambda_value"),
            lr=config.get("learning_rate"),
            metric=self.setup["fairness_metric"],
            n_epochs=epochs,
            on_test=False,
        )
        sub_path = CACHE_PATH / self.get_sub_path()
        file_name = f"test_{config.get('batch_size')}_{config.get('lambda_value')}_{config.get('learning_rate')}_{config.get('number_of_layers')}_{config.get('number_of_neurons_per_layer')}.txt"
        with open(sub_path / file_name, "w") as f:
            f.write("\n".join(map(str, predictions)))
        end_time = time.time()
        logger.debug(f"end training model")
        logger.debug(f"training time: {end_time - start_time}")
        test_y = test.iloc[:, -1].reset_index(drop=True)
        test_p = test.iloc[:, protected].reset_index(drop=True)
        test_singleton.append(conf_info | evaluate_predictions(test_p, predictions, test_y, logger))

        return results
