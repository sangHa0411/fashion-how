
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable

def estimate_fisher(dataset, _model, _device, batch_size):
    """
    estimate fisher information values
    """
    data_loader = dataset
    
    loglikelihoods = []
    
    cnt = 0
    for batch in data_loader:
        diag, cordi, rank = batch["diag"], batch["cordi"], batch["rank"]
        diag = diag.float().to(_device)
        cordi = cordi.long().to(_device)
        rank = rank.long().to(_device)

        logits = _model(diag, cordi)

        if (len(diag) == batch_size) and (cnt < 20):
            loglikelihoods.append(
                F.log_softmax(logits, dim=1)[range(batch_size), rank]
            )
        cnt = cnt+1 
    
    # estimate the fisher information of the parameters
    loglikelihoods = torch.cat(loglikelihoods).unbind()

    for i, l in enumerate(loglikelihoods, 1):
        loglikelihood_grads = zip(*[autograd.grad(l, _model.parameters(), retain_graph=(i < len(loglikelihoods)))])

    loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
    fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
    param_names = [n.replace('.', '__') for n, p in _model.named_parameters()]

    return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

def consolidate(fisher, _model):
    """
    save fisher information values for consolidation
    """
    for n, p in _model.named_parameters():
        n = n.replace('.', '__')        
        _model.register_buffer('{}_fisher'.format(n), fisher[n].data.clone())
                             
def ewc_loss(lamda, _model, cuda=False):
    """
    calculate ewc loss
    """
    try:
        losses = []
        for n, p in _model.named_parameters():
            # retrieve the consolidated mean and fisher information.
            n = n.replace('.', '__')

            mean = getattr(_model, '{}_mean'.format(n))
            fisher = getattr(_model, '{}_fisher'.format(n))
            # wrap mean and fisher in variables.
            mean = Variable(mean)
            fisher = Variable(fisher)

            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher * (p-mean)**2).sum())
        return (lamda/2)*sum(losses)
    except AttributeError:
        # ewc loss is 0 if there's no consolidated parameters.
        return (
            Variable(torch.zeros(1)).cuda() if cuda else
            Variable(torch.zeros(1))
        )
