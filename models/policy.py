

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        buf = eval_node[1:-1].split('][') # [6000,6000,6000,200][2000,2000]
        self._num_hid_eval = list(map(int, buf[0].split(','))) # 6000, 6000, 6000, 200
        self._num_hid_rnk = list(map(int, buf[1].split(',')))  # 2000, 2000
        self._num_hid_layer_eval = len(self._num_hid_eval) # 4
        self._num_hid_layer_rnk = len(self._num_hid_rnk)   # 2

        self.outer_embed = nn.Embedding(self._items_size[0], self._embed_size)
        self.top_embed = nn.Embedding(self._items_size[0], self._embed_size)
        self.bottom_embed = nn.Embedding(self._items_size[0], self._embed_size)
        self.shoes_embed = nn.Embedding(self._items_size[0], self._embed_size)

        mlp_eval_list = []
        num_in = self._embed_size * self._coordi_size + self._key_size
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
        # req : (batch_size, key_size) 
        # crd : (batch_size, num_rnk, coordi_size)
        crd_outer, crd_top, crd_bottom, crd_shoes = crd[:, :, 0], crd[:, :, 1], crd[:, :, 2], crd[:, :, 3]
        crd_outer_embed = self.outer_embed(crd_outer)     # (batch_size, num_rnk, embed_size)
        crd_top_embed = self.top_embed(crd_top)           # (batch_size, embed_size)
        crd_bottom_embed = self.bottom_embed(crd_bottom)  # (batch_size, embed_size)
        crd_shoes_embed = self.shoes_embed(crd_shoes)     # (batch_size, embed_size)

        crd_embed = torch.cat([crd_outer_embed, crd_top_embed, crd_bottom_embed, crd_shoes_embed], -1) # (batch_size, num_rnk, embed_size * cord_size)
        crd_embed_tr = torch.transpose(crd_embed, 1, 0) # (num_rnk, batch_size, embed_size * cord_size)
        
        in_rnks = []
        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_embed_tr[i], req) # (batch_size, eval_out_node)
            in_rnks.append(crd_eval)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)

        # (batch_size, num_rnk * eval_out_node + key_size)
        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)
        return out_rnk