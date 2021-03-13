import torch    
import torch.nn.functional as F

def IoU(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    y_true = (y_true == 1)
    y_pred = y_pred[:,1,:,:] > y_pred[:,1,:,:]
    eps = 1e-4
    intersection = (y_pred & y_true).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (y_pred | y_true).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
