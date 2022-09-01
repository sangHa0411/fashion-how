

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """Class for policy network"""
    def __init__(self, 
        embed_size,
        key_size, 
        dropout_prob,
        eval_node,
        num_rnk, 
        num_feature,
        num_cordi, 
        name='PolicyNet'
    ):
        """
        initialize and declare variables
        """
        super().__init__()
        self._embed_size = embed_size
        self._key_size = key_size
        self._dropout_prob = dropout_prob
        self._num_rnk = num_rnk
        self._num_feature = num_feature
        self._num_cordi = num_cordi
        self._name = name
        buf = eval_node[1:-1].split('][')
        self._num_hid_eval = list(map(int, buf[0].split(',')))
        self._num_hid_rnk = list(map(int, buf[1].split(',')))
        self._num_hid_layer_eval = len(self._num_hid_eval)
        self._num_hid_layer_rnk = len(self._num_hid_rnk)

        mlp_eval_list = []
        num_in = self._embed_size * self._num_feature + self._key_size
        for i in range(self._num_hid_layer_eval):
            num_out = self._num_hid_eval[i]
            sub_mlp_eval = nn.Sequential(
                nn.Linear(num_in, num_out),
                nn.ReLU(),
                nn.BatchNorm1d(num_out),
                nn.Dropout(self._dropout_prob)
            )
            mlp_eval_list.append(sub_mlp_eval) 
            num_in = num_out

        self._eval_out_node = num_out 
        self._mlp_eval = nn.Sequential(*mlp_eval_list)

        mlp_rnk_list = []
        num_in = self._eval_out_node * self._num_rnk + self._key_size
        for i in range(self._num_hid_layer_rnk):
            num_out = self._num_hid_rnk[i]
            sub_mlp_rnk = nn.Sequential(
                nn.Linear(num_in, num_out),
                nn.ReLU(),
                nn.BatchNorm1d(num_out),
                nn.Dropout(self._dropout_prob)
            )
            mlp_rnk_list.append(sub_mlp_rnk)
            num_in = num_out
        mlp_rnk_list.append(nn.Linear(num_in, self._num_rnk))
        self._mlp_rnk = nn.Sequential(*mlp_rnk_list) 

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """
        # (batch_size, num_feature, feature_size)
        crd_feature = torch.mean(crd, dim=1)

        # (batch_size, num_feature * feature_size)
        crd_tensor = [crd_feature[:, i, :] for i in range(self._num_cordi)]
        crd_tensor = torch.cat(crd_tensor, dim=-1) 

        crd_and_req = torch.cat((crd_tensor, req), 1) 
        evl = self._mlp_eval(crd_and_req)
        return evl
    
    def _ranking_coordi(self, in_rnk):
        """
        rank candidates         
        """
        out_rnk = self._mlp_rnk(in_rnk)
        return out_rnk
        
    def forward(self, req, crd):
        """
        build graph for evaluation and ranking         
        """
        # (num_rank, batch_size, coordi_size, num_feature, feature_size)
        crd_tr = torch.transpose(crd, 1, 0)
        
        in_rnks = []
        for i in range(self._num_rnk):
            # (batch_size, coordi_size, num_feature, feature_size)
            crd_eval = self._evaluate_coordi(crd_tr[i], req)
            in_rnks.append(crd_eval)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)

        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)
        return out_rnk