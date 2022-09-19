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

FOLD_SIZE = 3
NUM_RNK = 3
NUM_CORDI = 5
NUM_FEATURE = 4
IMG_FEAT_SIZE = 2048
KEY_SIZE = 512
TEXT_FEAT_SIZE = 512
DROPOUT_PROB = 0.1
MEM_SIZE = 24
HOPS = 3
NUM_LAYERS = 6
D_MODEL = 512
HIDDEN_SIZE = 2048
NUM_HEAD = 8
BATCH_SIZE = 8
EVAL_NODE = '[1024,4096,4096]'
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
    swer = SubWordEmbReaderUtil("/home/work/data/task4/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat")

    # -- Data
    print("\nLoading Dataset")
    eval_diag_loader = DialogueTestLoader("/home/work/data/task4/fs_eval_t1.wst.dev")
    eval_dataset = eval_diag_loader.get_dataset()

    # -- Encoding Data                    
    print("\nEncoding Dataset")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    encoder = Encoder(swer, num_cordi=NUM_CORDI, mem_size=MEM_SIZE)
    eval_encoded_dataset = encoder(eval_dataset)

    # -- Data Collator & Loader
    print("\nFormatting torch dataset")
    data_collator = PaddingCollator()
    eval_torch_dataset = FashionHowDataset(dataset=eval_encoded_dataset, 
        img_feat_dir="/home/work/data/task4/img_feats"
    )

    model_paths = [
        "/home/work/model/gAIa_model/gAIa-final-42.pt",
        "/home/work/model/gAIa_model/gAIa-final-1234.pt",
        "/home/work/model/gAIa_model/gAIa-final-95.pt"
    ]

    eval_predictions = []
    for i in range(FOLD_SIZE) :
        # -- Model
        model = Model(emb_size=swer.get_emb_size(),
            key_size=KEY_SIZE,
            mem_size=MEM_SIZE,
            hops=HOPS,
            eval_node=EVAL_NODE,
            num_rnk=NUM_RNK,
            num_feature=NUM_FEATURE,
            num_cordi=NUM_CORDI,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            hidden_size=HIDDEN_SIZE,
            num_head=NUM_HEAD,
            dropout_prob=DROPOUT_PROB,
            text_feat_size=TEXT_FEAT_SIZE,
            img_feat_size=IMG_FEAT_SIZE
        )
        model_path = model_paths[i]
        print("\nLoading %dth Model" %i)
        model.load_state_dict(torch.load(model_path, map_location=cuda_str))
        model.to(device)

        eval_dataloader = DataLoader(eval_torch_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=data_collator
        )

        # -- Inference
        print("%dth Inference" %i)
        predictions = []
        with torch.no_grad() :
            model.eval()
            for eval_data in eval_dataloader :
                diag, cordi = eval_data["diag"], eval_data["cordi"]
                diag = diag.float().to(device)
                cordi = cordi.float().to(device)
                
                logits = model(dlg=diag, crd=cordi)
                logits = logits.detach().cpu().numpy().tolist()

                predictions.extend(logits)
        eval_predictions.append(predictions)

    data_size = len(eval_predictions[0])
    print("\nThe number of test data : %d" %data_size)
    
    eval_ranks = []
    for i in range(data_size) :
        logits = [eval_predictions[j][i] for j in range(FOLD_SIZE)]
        logit = np.sum(logits, axis=0)

        pred = logit.argsort()[::-1]
        rank = [0, 0, 0]
        for j, p in enumerate(pred) :
            rank[p] = j
        eval_ranks.append(rank)

    print("\nPostprocessing predictions")
    eval_ids = []
    for rank in eval_ranks :
        eval_ids.append(convert(rank))

    eval_id_array = np.array(eval_ids)
    np.savetxt("/home/work/model/prediction.csv",
        eval_id_array.astype(int), 
        encoding='utf8', 
        fmt='%d'
    )

if __name__ == '__main__':
    inference()