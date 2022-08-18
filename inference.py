
import os
import torch
import argparse
from tqdm import tqdm
from scipy import stats
from torch.utils.data import DataLoader

from dataset.dataset import FashionHowDataset
from dataset.collator import PaddingCollator

from utils.encoder import Encoder
from utils.loader import MetaLoader, DialogueTestLoader

from models.model import Model
from models.tokenizer import SubWordEmbReaderUtil

def inference(args) :

    # -- Device
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    print("\nDevice:", device)

    # -- Subword Embedding
    print('\nInitializing subword embedding')
    swer = SubWordEmbReaderUtil(args.subWordEmb_path)

    # -- Meta Data
    print("\nLoading Meta Data...")
    meta_loader = MetaLoader(args.in_file_fashion, swer)
    img2id, _, _ = meta_loader.get_dataset()

    eval_diag_loader = DialogueTestLoader(args.in_file_tst_dialog, eval_flag=True)
    eval_dataset = eval_diag_loader.get_dataset()
    eval_rewards = [eval_data.pop("reward")for eval_data in eval_dataset]

    # -- Encoding Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    print("\nEncoding Data...")
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

    # -- Inference
    model.to(device)
    eval_predictions = []
    with torch.no_grad() :
        model.eval()
        for eval_data in tqdm(eval_dataloader) :
            diag, cordi = eval_data["diag"], eval_data["cordi"]
            diag = diag.float().to(device)
            cordi = cordi.long().to(device)
            logits = model(dlg=diag, crd=cordi)

            preds = torch.argsort(logits, -1).detach().cpu().numpy()
            eval_predictions.extend([pred.tolist()[::-1] for pred in preds])
    
    eval_tau = 0.0
    for i in range(len(eval_rewards)) :
        tau, _ = stats.weightedtau(eval_rewards[i], eval_predictions[i])
        eval_tau += tau

    eval_tau /= len(eval_rewards)
    print("Data : %s \t Evaluation Tau : %.4f" %(args.in_file_tst_dialog, eval_tau))

if __name__ == '__main__':

    # input options
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')
    parser.add_argument('--in_file_tst_dialog', type=str,
        default='./data/cl_eval_task1.wst.dev',
        help='test dialog DB'
    )
    parser.add_argument('--in_file_fashion', type=str,
        default='./data/mdata.wst.txt.2021.10.18',
        help='fashion item metadata'
    )
    parser.add_argument('--model_path', type=str,
        default='./gAIa_CL_model',
        help='path to save/read model'
    )
    parser.add_argument('--model_file', type=str,
        default=None,
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
        default=16,
        help='memory size for the MemN2N'
    )
    parser.add_argument('--key_size', type=int,
        default=300,
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
        default='[6000,6000,6000,200][2000,2000]',
        help='nodes of evaluation network'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )

    args = parser.parse_args()
    inference(args)
