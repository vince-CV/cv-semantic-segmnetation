import torch
import torch.nn as nn
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class SystemConfig:
    seed: int = 42                         
    cudnn_benchmark_enabled: bool = False  
    cudnn_deterministic: bool = True      

def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


class SoftDiceLoss(nn.Module):

    def __init__(self, num_classes, eps=1e-5):
        super().__init__()

        self.num_classes = num_classes
        self.eps = eps


    def forward(self, preds, targets):  
        """
            Compute Soft-Dice Loss.

            Arguments:
                preds (torch.FloatTensor): tensor of predicted labels. The shape of the tensor is (B, num_classes, H, W).
                targets (torch.LongTensor): tensor of ground-truth labels. The shape of the tensor is (B, 1, H, W).
            Returns:
                mean_loss (float32): mean loss by class  value.
        """
        loss = 0

        for cls in range(self.num_classes):

            target = (targets == cls).float()   # get ground truth for the current class

            pred = preds[:, cls]                # get prediction for the current class
            
            intersection = (pred * target).sum()               # calculate intersection

            dice = (2 * intersection + self.eps) / (pred.sum() + target.sum() + self.eps)             # compute dice coefficient

            loss = loss - dice.log()            # compute negative logarithm from the obtained dice coefficient

            loss = loss / self.num_classes      # get mean loss by class value

        return loss.item()


setup_system(SystemConfig)

ground_truth = torch.zeros(1, 224, 224)
ground_truth[:, :50, :50] = 1
ground_truth[:, 50:100, 50:100] = 2

# generate random predictions and use softmax to get probabilities
prediction = torch.zeros(1, 3, 224, 224).uniform_().softmax(dim=1)

# create an instance of a SoftDiceLoss class
soft_dice_loss = SoftDiceLoss(num_classes=3)


loss = soft_dice_loss(prediction, ground_truth)

print('Loss: {}'.format(loss))