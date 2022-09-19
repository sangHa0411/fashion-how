

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
        num_layers,
        d_model,
        hidden_size,
        num_head,
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
        self._num_hid_layer = num_layers
        self._d_model_hid = d_model
        self._nhead_hid = num_head
        self._hidden_hid = hidden_size
        self._name = name

        self._num_hid_rnk = list(map(int, eval_node[1:-1].split(',')))
        self._num_hid_layer_rnk = len(self._num_hid_rnk)

        self._query = nn.Linear(self._key_size, self._embed_size)

        mlp_eval_list = [nn.Linear(self._embed_size, self._d_model_hid)]
        hid_tf_layer = nn.TransformerEncoderLayer(d_model=self._d_model_hid, nhead=self._nhead_hid)
        hid_tf_encoder = nn.TransformerEncoder(hid_tf_layer, num_layers=self._num_hid_layer)
        mlp_eval_list.append(hid_tf_encoder)
        self._mlp_eval = nn.Sequential(*mlp_eval_list)

        mlp_rnk_list = []
        num_in = self._d_model_hid * self._num_rnk + self._key_size
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
        # crd (batch_size, num_cordi, num_feature, feature_size)
        crd_feature = torch.mean(crd, dim=1) # (batch_size, num_feature, feature_size)
        req = self._query(req) # (batch_size, feature_size)
        req = req.unsqueeze(1) # (batch_size, 1, feature_size)

        crd_and_req = torch.cat((crd_feature, req), 1) # (batch_size, num_feature+1, feature_size)
        evl = self._mlp_eval(crd_and_req)
        evl_tensor = torch.mean(evl, dim=1)
        return evl_tensor
    
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
            # (batch_size, feature_size)
            crd_eval = self._evaluate_coordi(crd_tr[i], req)
            in_rnks.append(crd_eval)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)

        # (batch_size, feature_size * _num_rnk + key_size)
        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)
        return out_rnk