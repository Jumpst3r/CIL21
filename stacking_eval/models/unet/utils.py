import torch
import torch.nn.functional as F


def F1(inputs, targets):
    inputs = torch.round(F.sigmoid(inputs))
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()

    return TP / (TP + 0.5 * (FP + FN))


def accuracy(inputs, targets):
    inputs = torch.round(F.sigmoid(inputs))
    T = (inputs == targets).sum()

    return T / targets.numel()


def IoU(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_true = y_true.type(torch.int32)
    y_pred = torch.round(F.sigmoid(y_pred)).type(torch.int32)
    y_pred = (y_pred == 1)
    y_true = (y_true == 1)
    eps = 1e-4
    intersection = (y_pred & y_true).float().sum((1, 2))
    union = (y_pred | y_true).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()
