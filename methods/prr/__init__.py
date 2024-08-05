import torch


class PRLoss:
    def __init__(self, eta=1.0, first_group_size: int = 0, second_group_size: int = 0):
        super(PRLoss, self).__init__()
        self.eta = eta
        self.first_group_size = first_group_size
        self.second_group_size = second_group_size

    def forward(self, first_group_output, second_group_output):
        # For the mutual information,
        # Pr[y|s] = sum{(xi,si),si=s} sigma(xi,si) / #D[xs]
        # D[xs]
        n_first_group = torch.tensor(first_group_output.shape[0])
        n_second_group = torch.tensor(second_group_output.shape[0])
        dxisi = torch.stack((n_second_group, n_first_group), axis=0)
        # Pr[y|s]
        y_pred_female = torch.sum(first_group_output)
        y_pred_male = torch.sum(second_group_output)
        p_ys = torch.stack((y_pred_male, y_pred_female), axis=0) / dxisi
        # Pr[y]
        p = torch.cat((first_group_output, second_group_output), 0)
        p_y = torch.sum(p) / (self.first_group_size + self.second_group_size)
        # P(siyi)
        p_s1y1 = torch.log(p_ys[1]) - torch.log(p_y)
        p_s1y0 = torch.log(1 - p_ys[1]) - torch.log(1 - p_y)
        p_s0y1 = torch.log(p_ys[0]) - torch.log(p_y)
        p_s0y0 = torch.log(1 - p_ys[0]) - torch.log(1 - p_y)
        # PI
        pi_s1y1 = first_group_output * p_s1y1
        pi_s1y0 = (1 - first_group_output) * p_s1y0
        pi_s0y1 = second_group_output * p_s0y1
        pi_s0y0 = (1 - second_group_output) * p_s0y0
        pi = torch.sum(pi_s1y1) + torch.sum(pi_s1y0) + torch.sum(pi_s0y1) + torch.sum(pi_s0y0)
        pi = self.eta * pi
        return pi


class PRLR:  # using linear
    def __init__(self, first_model, second_model, eta=0.0, epochs=3000, lr=0.01):
        super(PRLR, self).__init__()
        self.first_model = first_model
        self.second_model = second_model
        self.eta = eta
        self.epochs = epochs
        self.lr = lr

    def fit(self, x_female, y_female, x_male, y_male):
        criterion = torch.nn.BCELoss(reduction='sum')
        PI = PRLoss(eta=self.eta)
        epochs = self.epochs
        optimizer = torch.optim.Adam(list(self.first_model.parameters()) + list(self.second_model.parameters()), self.lr, weight_decay=1e-5)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output_f = self.first_model(x_female)
            output_m = self.second_model(x_male)
            logloss = criterion(output_f, y_female) + criterion(output_m, y_male)
            PIloss = PI.forward(output_f, output_m)
            loss = PIloss + logloss
            loss.backward()
            optimizer.step()
        self.first_model.eval()
        self.second_model.eval()
        # accu = accuracy(self.first_model, self.second_model, x_female, y_female, x_male, y_male)
        # cvs = CVS(self.first_model, self.second_model, x_female, x_male)
        # return accu, cvs
