import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


def F1(inputs, targets):
    inputs = torch.round(torch.sigmoid(inputs))
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()

    return TP / (TP + 0.5 * (FP + FN))


def IoU(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # import pdb; pdb.set_trace()
    y_true = y_true.type(torch.int32)
    y_pred = torch.round(torch.sigmoid(y_pred)).type(torch.int32)
    y_pred = (y_pred == 1)
    y_true = (y_true == 1)
    eps = 1e-4
    intersection = (y_pred & y_true).float().sum((1, 2))
    union = (y_pred | y_true).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() +
                                                        targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = F.sigmoid(inputs.reshape(-1))
        targets = targets.reshape(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.3, beta=0.5):
        ALPHA = 0.3  # < 0.5 penalises FP more, > 0.5 penalises FN more
        CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss
        eps = 1e-3
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() +
                                               smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = -(ALPHA * ((targets * torch.log(inputs)) +
                         ((1 - ALPHA) *
                          (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.2, beta=0.7, gamma=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky
