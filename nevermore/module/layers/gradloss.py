import torch.nn as nn
import torch


class GradLoss(nn.Module):
    def __init__(self, alpha, n_task):
        '''
        Args:
        alpha: float. 
        loss: list of loss
        '''

        super().__init__()

        self.alpha = alpha
        self.weights = torch.nn.Parameter(torch.ones(n_task))
        
    def forward(self, losses):
        total_losses = 0
        for idx,loss in enumerate(losses):
            total_losses += loss * self.weights[idx]
        return total_losses