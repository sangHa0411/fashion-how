import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.dataset import FashionHowDataset
from dataset.collator import PaddingCollator

from utils.encoder import Encoder
from utils.loader import DialogueTestLoader

from models.model import Model
from models.tokenizer import SubWordEmbReaderUtil

NUM_RNK = 3
NUM_CORDI = 5
NUM_FEATURE = 4
IMG_FEAT_SIZE = 2048
KEY_SIZE = 512
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
    print("\nSetting Device")
    cuda_flag = torch.cuda.is_available()
    cuda_str = "cuda" if cuda_flag else "cpu"
    device = torch.device(cuda_str)

    # -- Subword Embedding
    print("\nLoading Tokenizer")
    swer = SubWordEmbReaderUtil("data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat")

    # -- Data
    print("\nLoading Dataset")
    eval_diag_loader = DialogueTestLoader("data/fs_eval_t1.wst.dev")
    eval_dataset = eval_diag_loader.get_dataset()

    # -- Model
    print("\nLoading Model")
    model = Model(emb_size=swer.get_emb_size(),
        key_size=KEY_SIZE,
        mem_size=MEM_SIZE,
        hops=HOPS,
        eval_node=EVAL_NODE,
        num_rnk=NUM_RNK,
        num_feature=NUM_FEATURE,
        num_cordi=NUM_CORDI,
        dropout_prob=DROPOUT_PROB,
        text_feat_size=TEXT_FEAT_SIZE,
        img_feat_size=IMG_FEAT_SIZE
    )
    model_path = "gAIa_model/gAIa-final.pt"
    model.load_state_dict(torch.load(model_path, map_location=cuda_str))
    model.to(device)

    # -- Encoding Data                    
    print("\nEncoding Dataset")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    encoder = Encoder(swer, num_cordi=NUM_CORDI, mem_size=MEM_SIZE)
    eval_encoded_dataset = encoder(eval_dataset)

    # -- Data Collator & Loader
    print("\nFormatting torch dataset")
    data_collator = PaddingCollator()
    eval_torch_dataset = FashionHowDataset(dataset=eval_encoded_dataset, 
        img_feat_dir="/home/wkrtkd911/project/fashion-how/sub-task4/data/img_feats"
    )
    eval_dataloader = DataLoader(eval_torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=data_collator
    )

    # -- Inference
    print("\nInference")
    predictions = []
    with torch.no_grad() :
        model.eval()
        for eval_data in eval_dataloader :
            diag, cordi = eval_data["diag"], eval_data["cordi"]
            diag = diag.float().to(device)
            cordi = cordi.float().to(device)
            logits = model(dlg=diag, crd=cordi)

            preds = torch.argsort(logits, -1, descending=True).detach().cpu().numpy()
            ranks = []
            for pred in preds :
                rank = [0, 0, 0]
                for j, p in enumerate(pred) :
                    rank[p] = j
                ranks.append(rank)
            predictions.extend(ranks)

    pred_ids = []
    for pred in predictions :
        pred_ids.append(convert(pred))

    pred_id_array = np.array(pred_ids)
    np.savetxt(f"results/prediction.csv", 
        pred_id_array.astype(int), 
        encoding='utf8', 
        fmt='%d'
    )

if __name__ == '__main__':
    inference()