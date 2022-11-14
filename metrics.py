import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size()[1]

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice
