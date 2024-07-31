import time
import warnings
from ConfigSpace import Configuration
from sklearn.model_selection import KFold
from automl import MLP
from experiments import create_fully_connected_nn_tf, evaluate_predictions
from methods.fauci import create_fauci_network
from _logging import logger


class FauciMLP(MLP):
    def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:
        batch_size = config.get("batch_size")
        lambda_value = config.get("lambda_value")
        train = self.setup["train"]
        test = self.setup["test"]
        epochs = self.setup["epochs"]
        callbacks = self.setup["callbacks"]
        protected = self.setup["protected"]
        protected_type = self.setup["protected_type"]
        fairness_metric = self.setup["fairness_metric"]
        results = {}

        folds = KFold(n_splits=5, shuffle=True, random_state=seed)
        for exp_number, (train_idx, valid_idx) in enumerate(folds.split(train)):
            train_data, valid_data = train.iloc[train_idx], train.iloc[valid_idx]
            train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
            valid_x, valid_y = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
            test_x, _ = test.iloc[:, :-1], test.iloc[:, -1]

            # Create the model with FaUCI
            model = create_fully_connected_nn_tf(train.shape[1] - 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier = create_fauci_network(
                    model=model,
                    lambda_value=lambda_value,
                    protected_attribute=protected,
                    type_protected_attribute=protected_type,
                    fairness_metric=fairness_metric,
                )

                logger.debug(f"start training model")
                start_time = time.time()
                classifier.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                               validation_data=(valid_x, valid_y), verbose=0)
                end_time = time.time()
                logger.debug(f"end training model")
                logger.debug(f"training time: {end_time - start_time}")

                logger.debug(f"start predicting labels")
                predictions = classifier.predict(test_x)
                logger.debug(f"end predicting labels")

                test_y = test.iloc[:, -1].reset_index(drop=True)
                test_p = test.iloc[:, protected].reset_index(drop=True)
                tmp_results = evaluate_predictions(test_p, predictions, test_y, logger)
                # tmp_results["training_time"] = end_time - start_time
                results = {k: results.get(k, 0) + tmp_results.get(k, 0) for k in set(results) | set(tmp_results)}
        results = {k: v / 5 for k, v in results.items()}
        return results
