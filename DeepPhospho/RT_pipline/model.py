import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# pytorch < 1.4  (从pytorch 1.4里 copy出来的transformer的实现)


# pytorch >= 1.4
# from torch.nn import TransformerEncoderLayer, TransformerEncoder


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, pred_y):
        loss = torch.sqrt(self.mse(y, pred_y) + self.eps)
        return loss



