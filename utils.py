import torch    
import torch.nn.functional as F

def IoU(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    y_pred = (torch.round(F.sigmoid(y_pred)) == 1.)
    y_true = (y_true == 1)
    eps = 1e-4
    intersection = (y_pred & y_true).float().sum((1, 2))
    union = (y_pred | y_true).float().sum((1, 2))
    
    iou = (intersection + eps) / (union + eps)
        
    return iou.mean()