
import torch
from torch.nn import functional as F
from torch import autograd

def estimate_fisher(dataset, _model, _device, batch_size):
    """
    estimate fisher information values
    """
    data_loader = dataset
    
    loglikelihoods = []
    cnt = 0
    for batch in data_loader:
        dlg, crd, rnk = batch["diag"], batch["cordi"], batch["rank"]
        dlg = dlg.float().to(_device)
        crd = crd.long().to(_device)
        rnk = rnk.long().to(_device)

        logits = _model(dlg, crd)

        if (len(dlg) == batch_size) and (cnt < 20):
            loglikelihoods.append(
                F.log_softmax(logits, dim=1)[range(batch_size), rnk]
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
