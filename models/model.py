import torch
import torch.nn as nn 
from torch.autograd import Variable
from models.policy import PolicyNet
from models.requirement import RequirementNet

class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, 
        emb_size, 
        key_size, 
        mem_size,
        hops,
        item_sizes,
        coordi_size, 
        eval_node, 
        num_rnk,
        dropout_prob,
        img_feat_size,
        lamda=40,
        ):
        """
        initialize and declare variables
        """
        super().__init__()
        self.lamda = lamda
        # class instance for requirement estimation
        self._requirement = RequirementNet(emb_size, 
            key_size,
            mem_size, 
            hops
        )
        # class instance for ranking
        self._policy = PolicyNet(img_feat_size, 
            key_size, 
            coordi_size, 
            item_sizes,
            dropout_prob,
            eval_node,
            num_rnk, 
        )

        for name, param in self.named_parameters():
            n = name.replace('.', '__')
            self.register_buffer('{}_fisher'.format(n), torch.zeros(param.shape))

    def forward(self, dlg, crd):
        """
        build graph
        """
        # crd : (batch_size, num_rnk, coordi_size)
        req = self._requirement(dlg) # (batch_size, key_size)
        logits = self._policy(req, crd)
        return logits

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')        
            self.register_buffer('{}_fisher'.format(n), fisher[n].data.clone())
    
    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                mean = Variable(mean)
                fisher = Variable(fisher)

                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)[0]).cuda() if cuda else
                Variable(torch.zeros(1)[0])
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda