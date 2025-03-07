import torch

from analysis.metrics import Strategy

EPSILON: float = 1e-9
INFINITY: float = 1e9
DELTA: float = 5e-2  # percentage to apply to the values of the protected attribute to create the buckets


def single_conditional_probability(
        predicted,
        protected,
        value: int,
        equal: bool = True
):
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param value: the value of the protected attribute.
    @param equal: if True, filter rows whose protected attribute is equal to value, otherwise filter rows whose protected
    attribute is not equal to value.
    @return: the conditional probability.
    """
    if equal:
        rows = protected == value
    else:
        rows = protected != value
    return predicted[rows].mean()


def single_conditional_probability_in_range(
        predicted,
        protected,
        min_value: float,
        max_value: float,
        inside: bool = True,
):
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param min_value: the minimum value of the protected attribute.
    @param max_value: the maximum value of the protected attribute.
    attribute is not equal to value.
    @param inside: if True, filter rows whose protected attribute is inside the range, otherwise filter rows whose
    protected attribute is outside the range.
    @return: the conditional probability.
    """
    if inside:
        rows = (protected >= min_value) & (protected <= max_value)
    else:
        rows = (protected < min_value) | (protected > max_value)
    return predicted[rows].mean()


def double_conditional_probability(
        predicted,
        protected,
        ground_truth,
        first_value: int,
        second_value: int,
):
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param ground_truth: the ground truth.
    @param first_value: the value of the protected attribute.
    @param second_value: the value of the ground truth.
    @return: the conditional probability.
    :param equal:
    """
    rows = (protected == first_value) & (ground_truth == second_value)
    return predicted[rows].mean()


def double_conditional_probability_in_range(
        predicted,
        protected,
        ground_truth,
        min_value: float,
        max_value: float,
        second_value: int,
        inside: bool = True,
):
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param ground_truth: the ground truth.
    @param min_value: the minimum value of the protected attribute.
    @param max_value: the maximum value of the protected attribute.
    @param second_value: the value of the ground truth.
    attribute is not equal to value.
    @param inside: if True, filter rows whose protected attribute is inside the range, otherwise filter rows whose
    protected attribute is outside the range.
    @return: the conditional probability.
    """
    if inside:
        rows = (protected >= min_value) & (protected <= max_value) & (ground_truth == second_value)
    else:
        rows = (protected < min_value) | (protected > max_value) | (ground_truth != second_value)
    return predicted[rows].mean()


def discrete_demographic_parity(
        protected,
        y_pred,
        weights_strategy: int = Strategy.EQUAL
):
    """
    Calculate the demographic parity of a model.
    The protected attribute can be binary or categorical.

    @param protected: the protected attribute.
    @param y_pred: the predicted labels.
    @param weights_strategy: the strategy to compute the weights.
    @return: the demographic impact error.
    """
    unique_protected = protected.unique()
    absolute_probability = y_pred.mean()

    def _single_conditional_probability(value: int):
        return single_conditional_probability(y_pred, protected, value)

    probabilities = [_single_conditional_probability(value) for value in unique_protected]
    number_of_samples = [len(protected[protected == value]) for value in unique_protected]
    total_samples = sum(number_of_samples)
    if weights_strategy == Strategy.EQUAL:
        weights = 1 / len(unique_protected)
        return sum([abs(probability - absolute_probability) * weights for probability in probabilities])
    elif weights_strategy == Strategy.FREQUENCY:
        return sum([abs(probability - absolute_probability) * number_of_samples[i] / total_samples
                    for i, probability in enumerate(probabilities)])
    elif weights_strategy == Strategy.INVERSE_FREQUENCY:
        return sum([abs(probability - absolute_probability) * total_samples / number_of_samples[i]
                    for i, probability in enumerate(probabilities)])


def discrete_equalized_odds(
        protected,
        y_true,
        y_pred,
        weights_strategy: int = Strategy.EQUAL
):
    """
    Calculate the equalized odds of a model.
    The protected attribute can be binary or categorical.

    @param protected: the protected attribute.
    @param y_true: the ground truth.
    @param y_pred: the predicted labels.
    @param weights_strategy: the strategy to compute the weights.
    @return: the equalized odds error.
    """
    unique_protected = protected.unique()
    y_0 = (y_true == 0)
    y_1 = (y_true == 1)

    masks_a_y_0 = torch.tensor([double_conditional_probability(y_pred, protected, y_true, value, 0) for value in unique_protected]).to(y_pred.device)
    masks_a_y_1 = torch.tensor([double_conditional_probability(y_pred, protected, y_true, value, 1) for value in unique_protected]).to(y_pred.device)

    mask_y_0 = y_pred[y_0].float().mean()
    mask_y_1 = y_pred[y_1].float().mean()

    number_of_samples_a_0 = torch.tensor([((protected == value) & y_0).sum().float() for value in unique_protected]).to(y_pred.device)
    number_of_samples_a_1 = torch.tensor([((protected == value) & y_1).sum().float() for value in unique_protected]).to(y_pred.device)

    differences_0 = torch.abs(masks_a_y_0 - mask_y_0)
    differences_1 = torch.abs(masks_a_y_1 - mask_y_1)

    total_samples = number_of_samples_a_0.sum() + number_of_samples_a_1.sum()
    number_unique = unique_protected.numel()

    if weights_strategy == Strategy.EQUAL:
        return (differences_0.sum() + differences_1.sum()) / (2 * number_unique)
    else:
        a_occurrences = torch.tensor([(protected == value).sum().float() for value in unique_protected]).to(y_pred.device)
        if weights_strategy == Strategy.FREQUENCY:
            differences_0 *= number_of_samples_a_0
            differences_1 *= number_of_samples_a_1
            return (differences_0.sum() + differences_1.sum()) * a_occurrences / total_samples
        elif weights_strategy == Strategy.INVERSE_FREQUENCY:
            differences_0 *= (total_samples - number_of_samples_a_0)
            differences_1 *= (total_samples - number_of_samples_a_1)
            return (differences_0.sum() + differences_1.sum()) * (1 - a_occurrences / total_samples) / (number_unique - 1)
