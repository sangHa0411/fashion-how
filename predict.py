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

def inference(args) :

    # -- Device
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")

    # -- Subword Embedding
    swer = SubWordEmbReaderUtil(args.subWordEmb_path)

    # -- Meta Data
    meta_loader = MetaLoader(args.in_file_fashion, swer)
    img2id, _, _ = meta_loader.get_dataset()

    in_file_tst_dialog = ["data/test/cl_eval_task1.wst.tst", 
        "data/test/cl_eval_task2.wst.tst", 
        "data/test/cl_eval_task3.wst.tst",
        "data/test/cl_eval_task4.wst.tst",
        "data/test/cl_eval_task5.wst.tst",
        "data/test/cl_eval_task6.wst.tst"
    ]

    # -- Model
    num_rnk = 3
    coordi_size = 4
    item_sizes = [len(img2id[i]) for i in range(4)]
    model = Model(emb_size=swer.get_emb_size(),
        key_size=args.key_size,
        mem_size=args.mem_size,
        hops=args.hops,
        item_sizes=item_sizes,
        coordi_size=coordi_size,
        eval_node=args.eval_node,
        num_rnk=num_rnk,
        dropout_prob=args.dropout_prob,
        text_feat_size=args.text_feat_size,
        img_feat_size=args.img_feat_size
    )
    if args.model_file is not None:
        model_path = os.path.join(args.model_path, args.model_file)
        model.load_state_dict(torch.load(model_path))
        print("\nLoaded model from %s" %model_path)


    model.to(device)
    eval_predictions = []
    for i in range(len(in_file_tst_dialog)) :
        file_tst_dialog = in_file_tst_dialog[i]

        eval_diag_loader = DialogueTestLoader(file_tst_dialog, eval_flag=False)
        eval_dataset = eval_diag_loader.get_dataset()

        # -- Encoding Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        encoder = Encoder(swer, img2id, num_cordi=4, mem_size=args.mem_size)
        eval_encoded_dataset = encoder(eval_dataset)

        # -- Data Collator & Loader
        data_collator = PaddingCollator()
        eval_torch_dataset = FashionHowDataset(dataset=eval_encoded_dataset)
        eval_dataloader = DataLoader(eval_torch_dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
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
    np.savetxt(f"./predictions/prediction.csv", 
        predictions.astype(int), 
        encoding='utf8', 
        fmt='%d'
    )

if __name__ == '__main__':

    # input options
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')
    parser.add_argument('--in_file_fashion', type=str,
        default='./data/mdata.wst.txt.2021.10.18',
        help='fashion item metadata'
    )
    parser.add_argument('--model_path', type=str,
        default='./gAIa_CL_model',
        help='path to save/read model'
    )
    parser.add_argument('--model_file', type=str,
        default="gAIa-final.pt",
        help='model file name'
    )
    parser.add_argument('--subWordEmb_path', type=str,
        default='./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat',
        help='path of subword embedding'
    )
    parser.add_argument('--dropout_prob', type=float,
        default=0.1,
        help='dropout prob.'
    )
    parser.add_argument('--eval_batch_size', type=int,
        default=16,
        help='batch size for training'
    )
    parser.add_argument('--hops', type=int,
        default=3,
        help='number of hops in the MemN2N'
    )
    parser.add_argument('--mem_size', type=int,
        default=24,
        help='memory size for the MemN2N'
    )
    parser.add_argument('--key_size', type=int,
        default=512,
        help='memory size for the MemN2N'
    )
    parser.add_argument('--img_feat_size', type=int,
        default=512,
        help='size of image feature'
    )
    parser.add_argument('--text_feat_size', type=int,
        default=512,
        help='size of text feature'
    )
    parser.add_argument('--eval_node', type=str,
        default='[6000,6000,6000,6000,512][2000,2000,2000]',
        help='nodes of evaluation network'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )

    args = parser.parse_args()
    inference(args)
