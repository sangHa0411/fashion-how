import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from dataset.dataset import FashionHowDataset
from dataset.collator import PaddingCollator

from utils.encoder import Encoder
from utils.loader import MetaLoader, DialogueTestLoader

from models.model import Model
from models.tokenizer import SubWordEmbReaderUtil

KEY_SIZE = 512
IMG_FEAT_SIZE = 512
TEXT_FEAT_SIZE = 512
DROPOUT_PROB = 0.1
MEM_SIZE = 24
HOPS = 3
BATCH_SIZE = 8
EVAL_NODE = '[6000,6000,6000,6000,512][2000,2000,2000]'
NUM_WORKERS = 2

def convert(pred) :
    if pred[0] == 0 and pred[1] == 1 and pred[2] == 2 :
        return 0
    elif pred[0] == 0 and pred[1] == 2 and pred[2] == 1 :
        return 1
    elif pred[0] == 1 and pred[1] == 0 and pred[2] == 2 :
        return 2
    elif pred[0] == 1 and pred[1] == 2 and pred[2] == 0 :
        return 3
    elif pred[0] == 2 and pred[1] == 0 and pred[2] == 1 :
        return 4
    elif pred[0] == 2 and pred[1] == 1 and pred[2] == 0 :
        return 5
    else :
        return -1

def inference() :

    # -- Device
    cuda_flag = torch.cuda.is_available()
    cuda_str = "cuda" if cuda_flag else "cpu"
    device = torch.device(cuda_str)

    # -- Subword Embedding
    swer = SubWordEmbReaderUtil("/home/work/data/task3/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat")

    # -- Meta Data
    meta_loader = MetaLoader("/home/work/model/mdata.wst.txt.2021.10.18", swer)
    img2id, _, _ = meta_loader.get_dataset()
    
    in_file_tst_dialog = [
        "/home/work/data/task3/cl_eval_task1.wst.tst", 
        "/home/work/data/task3/cl_eval_task2.wst.tst", 
        "/home/work/data/task3/cl_eval_task3.wst.tst",
        "/home/work/data/task3/cl_eval_task4.wst.tst",
        "/home/work/data/task3/cl_eval_task5.wst.tst",
        "/home/work/data/task3/cl_eval_task6.wst.tst"
    ]

    # -- Model
    num_rnk = 3
    coordi_size = 4
    item_sizes = [len(img2id[i]) for i in range(4)]
    model = Model(emb_size=swer.get_emb_size(),
        key_size=KEY_SIZE,
        mem_size=MEM_SIZE,
        hops=HOPS,
        item_sizes=item_sizes,
        coordi_size=coordi_size,
        eval_node=EVAL_NODE,
        num_rnk=num_rnk,
        dropout_prob=DROPOUT_PROB,
        text_feat_size=TEXT_FEAT_SIZE,
        img_feat_size=IMG_FEAT_SIZE
    )
    model_path = "gAIa_CL_model/gAIa-final.pt"
    model.load_state_dict(torch.load(model_path, map_location=cuda_str))

    model.to(device)
    eval_predictions = []
    for i in range(len(in_file_tst_dialog)) :
        file_tst_dialog = in_file_tst_dialog[i]

        eval_diag_loader = DialogueTestLoader(file_tst_dialog, eval_flag=False)
        eval_dataset = eval_diag_loader.get_dataset()

        # -- Encoding Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        encoder = Encoder(swer, img2id, num_cordi=4, mem_size=MEM_SIZE)
        eval_encoded_dataset = encoder(eval_dataset)

        # -- Data Collator & Loader
        data_collator = PaddingCollator()
        eval_torch_dataset = FashionHowDataset(dataset=eval_encoded_dataset)
        eval_dataloader = DataLoader(eval_torch_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=data_collator
        )

        # -- Inference
        with torch.no_grad() :
            model.eval()
            for eval_data in eval_dataloader :
                diag, cordi = eval_data["diag"], eval_data["cordi"]
                diag = diag.float().to(device)
                cordi = cordi.long().to(device)
                logits = model(dlg=diag, crd=cordi)

                preds = torch.argsort(logits, -1, descending=True).detach().cpu().numpy()
                ranks = []
                for pred in preds :
                    rank = [0, 0, 0]
                    for j, p in enumerate(pred) :
                        rank[p] = j
                    ranks.append(rank)
                eval_predictions.extend(ranks)

    predictions = []
    for pred in eval_predictions :
        predictions.append(convert(pred))

    predictions = np.array(predictions)
    np.savetxt(f"/home/work/model/prediction.csv", 
        predictions.astype(int), 
        encoding='utf8', 
        fmt='%d'
    )

if __name__ == '__main__':
    inference()
