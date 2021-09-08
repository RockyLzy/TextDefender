import numpy as np
import torch
import torch.nn as nn


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, lr=1e-3, beta=0):
        super(CLUB, self).__init__()
        self.hiddensize = y_dim
        self.version = 2
        self.beta = beta

    def mi_est_sample(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        positive = torch.zeros_like(y_samples)

        prediction_1 = y_samples.unsqueeze(1)  # [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean(), positive.sum(dim = -1).mean(), negative.sum(dim = -1).mean()

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, x_samples, y_samples, steps=None):
        # no performance improvement, not enabled
        if steps:
            beta = self.beta if steps > 1000 else self.beta * steps / 1000  # beta anealing
        else:
            beta = self.beta

        return self.mi_est_sample(x_samples, y_samples) * self.beta


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(InfoNCE, self).__init__()
        self.lower_size = 300
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, self.lower_size),
                                    nn.ReLU(),
                                    nn.Linear(self.lower_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [s_size, s_size, 1]

        lower_bound = T0.mean() - (
                    T1.logsumexp(dim=1).mean() - np.log(sample_size))  # torch.log(T1.exp().mean(dim = 1)).mean()

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound