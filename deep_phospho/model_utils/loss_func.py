import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_y, y):
        pred_y = pred_y.reshape(pred_y.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        pred_y_zero_mean = pred_y - torch.mean(pred_y, 1).reshape(pred_y.shape[0], 1)
        y_zero_mean = y - torch.mean(y, 1).reshape(y.shape[0], 1)
        # ipdb.set_trace()
        assert torch.all(torch.sqrt(torch.sum(pred_y_zero_mean ** 2, 1)) != 0)
        assert torch.all(torch.sqrt(torch.sum(y_zero_mean ** 2, 1)) != 0)
        pcc = torch.sum(pred_y_zero_mean * y_zero_mean, 1) / (torch.sqrt(torch.sum(pred_y_zero_mean ** 2, 1)) *
                                                              torch.sqrt(torch.sum(y_zero_mean ** 2, 1)))

        loss = 1 - pcc
        return torch.mean(loss + self.eps)


class SALoss(nn.Module):

    def __init__(self, eps=1e-6):
        super(SALoss, self).__init__()
        self.eps = eps

    def forward(self, pred_y, y):
        pred_y = pred_y.reshape(pred_y.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        pred_y_l2norm = torch.norm(pred_y, dim=1).reshape(pred_y.shape[0], 1)
        y_l2norm = torch.norm(y, dim=1).reshape(y.shape[0], 1)
        # assert pred_y_l2norm != 0
        # assert y_l2norm != 0
        pred_y_l2normed = pred_y / pred_y_l2norm

        y_l2normed = y / y_l2norm

        loss = 2 * torch.acos(torch.sum(pred_y_l2normed * y_l2normed, 1)) / np.pi
        # ipdb.set_trace()
        return torch.mean(loss + self.eps)


class SALoss_MSE(nn.Module):

    def __init__(self):
        super(SALoss_MSE, self).__init__()

    def forward(self, pred_y, y):
        sa_loss = SALoss()
        mse_loss = nn.MSELoss()
        sa_part = sa_loss(pred_y, y)
        mse_part = mse_loss(pred_y, y)

        return torch.mean(sa_part + mse_part)


class SA_Pearson_Loss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_y, y):
        sa_loss = SALoss()
        sa_loss_part = sa_loss(pred_y, y)
        pc_loss = PearsonLoss()
        pc_loss_part = pc_loss(pred_y, y)

        return torch.mean(sa_loss_part + pc_loss_part + self.eps)


class L1_SA_Pearson_Loss(nn.Module):

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred_y, y):
        sa_loss = SALoss()
        sa_loss_part = sa_loss(pred_y, y)
        pc_loss = PearsonLoss()
        pc_loss_part = pc_loss(pred_y, y)
        l1_loss = nn.L1Loss()
        l1_loss_part = l1_loss(pred_y, y)

        return torch.mean(sa_loss_part + pc_loss_part + l1_loss_part + self.eps)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, pred_y):
        loss = torch.sqrt(self.mse(y, pred_y) + self.eps)
        return loss


class MaskedLanguageLoss(nn.Module):
    def __init__(self, only_acc_masked_token=False):
        super().__init__()
        self.only_acc_masked_token = only_acc_masked_token

        self.celoss = nn.CrossEntropyLoss(ignore_index=0,
                                          reduction='mean')
        if not only_acc_masked_token:
            # because the padding tokens are so much, which could lead to imbalanced data
            # here we add a weight for padding token
            weights = torch.ones(31, device='cuda')
            weights *= 0.9
            weights[0] = 0.1
            self.celoss = nn.CrossEntropyLoss(weight=weights,
                                              reduction='mean')

    def forward(self, pred_y, y):
        labels = y[0]
        mask = y[1]
        if self.only_acc_masked_token:
            labels = (labels * mask).long().cuda()

        # see prediction for debugging
        # print(F.softmax(pred_y, dim=-1).max(dim=-1)[-1][:8, :15])
        # print(labels[:8, :15])
        # print()
        loss = self.celoss(pred_y.transpose(1, 2), labels)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
