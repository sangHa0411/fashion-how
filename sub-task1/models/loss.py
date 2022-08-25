
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module) :

    def __init__(self,) :
        super(CrossEntropyLoss, self).__init__()
        pass

    def __call__(self, logits, labels) :

        log_softmax = -F.log_softmax(logits, dim=-1)
        loss = log_softmax * labels
        loss_per_data = torch.mean(loss, dim=-1)
        mean_loss = torch.mean(loss_per_data)
        return mean_loss

