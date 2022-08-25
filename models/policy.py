

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module) :

    def __init__(self, num_in, num_hidden, num_out, dropout_prob) :
        super().__init__()
        self._num_in = num_in
        self._num_hidden = num_hidden
        self._num_out = num_out
        self._dropout_prob = dropout_prob

        self._net1 = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_out),
        )
        self._net2 = None if num_in == num_out else \
            nn.Linear(num_in, num_out)
        self._drop = nn.Dropout(dropout_prob)
        self._norm = nn.BatchNorm1d(num_out)

    def forward(self, i_tensor) :
        h_tensor = self._net1(i_tensor)
        h_tensor = self._drop(h_tensor)

        if self._net2 is not None :
            i_tensor = self._net2(i_tensor)

        o_tensor = self._norm(h_tensor + i_tensor)
        return o_tensor


class PolicyNet(nn.Module):
    """Class for policy network"""
    def __init__(self, 
        embed_size,
        key_size, 
        coordi_size, 
        item_sizes,
        dropout_prob,
        eval_node,
        num_rnk, 
        name='PolicyNet'
    ):
        """
        initialize and declare variables
        """
        super().__init__()
        self._embed_size = embed_size
        self._key_size = key_size
        self._coordi_size = coordi_size
        self._dropout_prob = dropout_prob
        self._num_rnk = num_rnk
        self._name = name
        self._items_size = item_sizes
        buf = eval_node[1:-1].split('][')
        self._num_hid_eval = list(map(int, buf[0].split(',')))
        self._num_hid_rnk = list(map(int, buf[1].split(',')))
        self._num_hid_layer_eval = len(self._num_hid_eval)
        self._num_hid_layer_rnk = len(self._num_hid_rnk)

        self.outer_embed = nn.Embedding(self._items_size[0], self._embed_size)
        self.top_embed = nn.Embedding(self._items_size[1], self._embed_size)
        self.bottom_embed = nn.Embedding(self._items_size[2], self._embed_size)
        self.shoes_embed = nn.Embedding(self._items_size[3], self._embed_size)
        
        self.outer_bias = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
            size=(1, self._embed_size)), 
            requires_grad=True
            )
        self.top_bias = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
            size=(1, self._embed_size)), 
            requires_grad=True
        )
        self.bottom_bias = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
            size=(1, self._embed_size)), 
            requires_grad=True
        )
        self.shoes_bias = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
            size=(1, self._embed_size)), 
            requires_grad=True
        )
        
        mlp_eval_list = []
        num_in = self._embed_size * self._coordi_size + self._key_size
        for i in range(self._num_hid_layer_eval):
            num_out = self._num_hid_eval[i]
            sub_mlp_eval = nn.Sequential(
                nn.Linear(num_in, num_out),
                nn.ReLU(),
                nn.BatchNorm1d(num_out),
                nn.Dropout(dropout_prob)
            )
            mlp_eval_list.append(sub_mlp_eval) 
            num_in = num_out

        self._eval_out_node = num_out 
        self._mlp_eval = nn.Sequential(*mlp_eval_list)

        mlp_rnk_list = []
        num_in = self._eval_out_node * self._num_rnk + self._key_size
        for i in range(self._num_hid_layer_rnk):
            num_out = self._num_hid_rnk[i]
            sub_mlp_rnk = FeedForward(num_in, 4096, num_out, dropout_prob)
            # sub_mlp_rnk = nn.Sequential(
            #     nn.Linear(num_in, num_out),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(num_out),
            #     nn.Dropout(dropout_prob)
            # )
            mlp_rnk_list.append(sub_mlp_rnk)
            num_in = num_out
        mlp_rnk_list.append(nn.Linear(num_in, self._num_rnk))
        self._mlp_rnk = nn.Sequential(*mlp_rnk_list) 

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """
        crd_and_req = torch.cat((crd, req), 1)
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
        crd_outer, crd_top, crd_bottom, crd_shoes = crd[:, :, 0], crd[:, :, 1], crd[:, :, 2], crd[:, :, 3]
        crd_outer_embed = self.outer_embed(crd_outer) + self.outer_bias
        crd_top_embed = self.top_embed(crd_top) + self.top_bias
        crd_bottom_embed = self.bottom_embed(crd_bottom) + self.bottom_bias
        crd_shoes_embed = self.shoes_embed(crd_shoes) + self.shoes_bias

        crd_embed = torch.cat([crd_outer_embed, crd_top_embed, crd_bottom_embed, crd_shoes_embed], -1) 
        crd_embed_tr = torch.transpose(crd_embed, 1, 0)
        
        in_rnks = []
        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_embed_tr[i], req)
            in_rnks.append(crd_eval)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)

        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)
        return out_rnk