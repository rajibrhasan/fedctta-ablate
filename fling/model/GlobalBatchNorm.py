import torch
import torch.nn as nn

class GlobalBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, ratio=0.5):
        super(GlobalBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.sum_num = 0.
        self.sample_num = 0.
        self.ratio = ratio
        self.bank_mean = self.running_mean.view(1, -1)
        self.bank_var = self.running_var.view(1, -1)
        self.weighted_mean, self.weighted_var = [], []

    def update(self, mean, var):
        self.mean = mean
        self.var = var

    def update_sample_num(self, sample_num):
        self.sample_num = sample_num

    def forward(self, input):                             # input(N,C,H,W)
        if self.training:
            self.batch_mean = input.mean([0, 2, 3])
            self.batch_var = input.var([0, 2, 3], unbiased=False)
            self.mean = self.batch_mean
            self.var = self.batch_var
        else:
            if self.weighted_mean == []:
                self.batch_mean = input.mean([0, 2, 3])
                self.batch_var = input.var([0, 2, 3], unbiased=False)
                self.mean = self.batch_mean
                self.var = self.batch_var
            else:
                self.mean = self.weighted_mean
                self.var = self.weighted_var

        input = self.alpha * (input - self.mean[None, :, None, None]) / (torch.sqrt(self.var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


    def update_prior(self, sample_num):
        if self.past_mean == 0:
            self.past_mean = self.batch_mean
            self.past_var = self.batch_var
        else:
            self.past_mean = (self.past_mean * self.sum + self.mean * sample_num) / (self.sum + sample_num)
            self.past_var = (self.past_var * self.sum + self.var * sample_num) / (self.sum + sample_num)
        self.sum += sample_num

