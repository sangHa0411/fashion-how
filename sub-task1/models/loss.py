
import torch
import torch.nn.functional as F

def loss_fn(logit, label) :

    log_softmax = -F.log_softmax(logit, dim=-1)
    loss = log_softmax * label
    loss_per_data = torch.mean(loss, dim=-1)
    mean_loss = torch.mean(loss_per_data)
    return mean_loss

def acc_fn(logit, label) :

    acc = 0.0
    logit = logit.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    for j in range(len(logit)) :
        if logit[j].argmax() == label[j].argmax() :
            acc += 1.0

    return acc