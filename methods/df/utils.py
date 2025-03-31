import numpy as np
import torch

# %%
# Measure intersectional DF from positive predict probabilities
def differential_fairness_binary_outcome(positive_probs):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilon_per_group = np.zeros(len(positive_probs))
    for i in range(len(positive_probs)):
        epsilon = 0.0  # initialization of DF
        for j in range(len(positive_probs)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon, abs(np.log(positive_probs[i]) - np.log(
                    positive_probs[j])))  # ratio of probabilities of positive outcome
                epsilon = max(epsilon, abs(np.log((1 - positive_probs[i])) - np.log(
                    (1 - positive_probs[j]))))  # ratio of probabilities of negative outcome
        epsilon_per_group[i] = epsilon  # DF per group
    epsilon = max(epsilon_per_group)  # overall DF of the algorithm
    return epsilon


# %%
# Measure SP-Subgroup fairness (gamma unfairness)
def subgroup_fairness(positive_probs, alpha_sp):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    sp_d = sum(positive_probs * alpha_sp)  # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gamma_per_group = np.zeros(len(positive_probs))  # SF per group
    for i in range(len(positive_probs)):
        gamma_per_group[i] = alpha_sp[i] * abs(sp_d - positive_probs[i])
    gamma = max(gamma_per_group)  # overall SF of the algorithm
    return gamma


# %%
# smoothed empirical differential fairness measurement
def compute_edf_on_data(protected, predictions, predict_prob, subgroups):
    # compute counts and probabilities
    counts_class_one = np.zeros(len(subgroups))
    counts_total = np.zeros(len(subgroups))
    counts_class_one_soft = np.zeros(len(subgroups))

    num_classes = 2
    concentration_parameter = 1.0
    dirichlet_alpha = concentration_parameter / num_classes
    x_control = np.int64(np.ones((len(predictions))))

    for i in range(len(predictions)):
        index = np.where((subgroups == protected[i]).all(axis=1))[0][0]
        counts_total[index] = counts_total[index] + 1
        counts_class_one_soft[index] = counts_class_one_soft[index] + predict_prob[i]
        if predictions[i] == 1:
            counts_class_one[index] = counts_class_one[index] + 1
        if protected[i, 0] == 0 and protected[i, 1] == 0:
            x_control[i] = 0

    # probability of y given S (p(y=1|S)): probability distribution over merit per value of the protected attributes
    probabilities_of_positive_hard = (counts_class_one + dirichlet_alpha) / (counts_total + concentration_parameter)
    epsilon = differential_fairness_binary_outcome(probabilities_of_positive_hard)

    return epsilon


# Loss and optimizer
def fairness_loss(base_fairness, countClass_hat, countTotal_hat):
    # DF-based penalty term
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    zeroTerm = torch.tensor(0.0)

    theta = (countClass_hat + dirichletAlpha) / (countTotal_hat + concentrationParameter)
    # theta = theta/sum(theta)
    epsilonClass = df_binary_outcome_train(theta)
    return torch.max(zeroTerm, (epsilonClass - base_fairness))


# Loss and optimizer
def sf_loss(base_fairness, countClass_hat, countTotal_hat):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    zeroTerm = torch.tensor(0.0)
    population = sum(countTotal_hat).detach()

    theta = (countClass_hat + dirichletAlpha) / (countTotal_hat + concentrationParameter)
    alpha = (countTotal_hat + dirichletAlpha) / (population + concentrationParameter)
    # theta = theta/sum(theta)
    gammaClass = subgroup_fairness_train(theta, alpha)
    return torch.max(zeroTerm, (gammaClass - base_fairness))


def prule_loss(base_fairness, countClass_hat, countTotal_hat):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    zeroTerm = torch.tensor(0.0)

    theta_minority = (countClass_hat[0] + dirichletAlpha) / (countTotal_hat[0] + concentrationParameter)
    theta_majority = (countClass_hat[1] + dirichletAlpha) / (countTotal_hat[1] + concentrationParameter)
    # theta = theta/sum(theta)
    pruleClass = torch.min(theta_minority / theta_majority, theta_majority / theta_minority) * 100.0
    return torch.max(zeroTerm, (base_fairness - pruleClass))


# %%
# Measure intersectional DF from positive predict probabilities
def df_binary_outcome_train(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(len(probabilitiesOfPositive), dtype=torch.float)
    for i in range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0)  # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = torch.max(epsilon, torch.abs(torch.log(probabilitiesOfPositive[i]) - torch.log(
                    probabilitiesOfPositive[j])))  # ratio of probabilities of positive outcome
                epsilon = torch.max(epsilon, torch.abs((torch.log(1 - probabilitiesOfPositive[i])) - (
                    torch.log(1 - probabilitiesOfPositive[j]))))  # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon  # DF per group
    epsilon = torch.max(epsilonPerGroup)  # overall DF of the algorithm
    return epsilon


def subgroup_fairness_train(probabilitiesOfPositive, alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(
        probabilitiesOfPositive * alphaSP)  # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = torch.zeros(len(probabilitiesOfPositive), dtype=torch.float)  # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i] * torch.abs(spD - probabilitiesOfPositive[i])
    gamma = torch.max(gammaPerGroup)  # overall SF of the algorithm
    return gamma


# %% empirical count on train set
def compute_batch_counts(protectedAttributes, intersectGroups, predictions):
    matches = (protectedAttributes[:, None, :] == intersectGroups).all(dim=2)
    indices = torch.nonzero(matches, as_tuple=True)[1]

    countsTotal = torch.bincount(indices, minlength=len(intersectGroups)).to(torch.float)

    countsClassOne = torch.zeros_like(countsTotal)
    countsClassOne.scatter_add_(0, indices, torch.squeeze(predictions))

    return countsClassOne, countsTotal
