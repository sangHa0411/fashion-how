
import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import FashionHowDataset
from dataset.collator import PaddingCollator

from utils.encoder import Encoder
from utils.augmentation import DataAugmentation
from utils.preprocessor import DiagPreprocessor
from utils.loader import MetaLoader, DialogueTrainLoader

from models.model import Model
from models.scheduler import LinearWarmupScheduler
from models.tokenizer import SubWordEmbReaderUtil

NUM_RNK = 3
NUM_CORDI = 4
NUM_FEATURE = 4
IMG_FEAT_SIZE = 2048

def train(args) :

    print("\nTraining Data : %s" %args.in_file_trn_dialog)

    # -- Seed
    seed_everything(args.seed)

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
    img2id, id2img, img_similarity = meta_loader.get_dataset()

    # -- Dialogue Data
    print("\nLoading Dialogue Data...")
    train_diag_loader = DialogueTrainLoader(args.in_file_trn_dialog)
    train_raw_dataset = train_diag_loader.get_dataset()

    # -- Dialogue Preprocessor
    print("\nPreprocessing Dialogue Data...")
    diag_preprocessor = DiagPreprocessor(num_rank=3, num_cordi=4)
    train_dataset = diag_preprocessor(train_raw_dataset, img2id, id2img, img_similarity)
    print("The number of train dataset : %d" %len(train_dataset))

    # -- Data Augmentation
    print("\nData Augmentation...")
    data_augmentation = DataAugmentation(num_aug=args.augmentation_size)
    train_dataset = data_augmentation(train_dataset, img2id, id2img, img_similarity)
    print("The number of train dataset : %d" %len(train_dataset))

    # -- Encoding Data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    print("\nEncoding Data...")
    encoder = Encoder(swer, num_cordi=4, mem_size=args.mem_size)
    train_encoded_dataset = encoder(train_dataset)

    # -- Data Collator & Loader
    data_collator = PaddingCollator()
    train_torch_dataset = FashionHowDataset(dataset=train_encoded_dataset, img_feat_dir=args.img_feat_dir)
    train_dataloader = DataLoader(train_torch_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )

    # -- Model
    model = Model(emb_size=swer.get_emb_size(),
        key_size=args.key_size,
        mem_size=args.mem_size,
        hops=args.hops,
        eval_node=args.eval_node,
        num_rnk=NUM_RNK,
        num_feature=NUM_FEATURE,
        num_cordi=NUM_CORDI,
        num_layers=args.num_layers,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_head=args.num_head,
        dropout_prob=args.dropout_prob,
        text_feat_size=args.text_feat_size,
        img_feat_size=IMG_FEAT_SIZE
    )
    
    # -- Optimizer, Loss Function
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    loss_ce = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LinearWarmupScheduler(optimizer, total_steps, warmup_steps)
    
    # -- Training
    acc = 0.0
    model.to(device)

    print("\nTraining model")
    train_data_iterator = iter(train_dataloader)
    for step in tqdm(range(total_steps)) :
        try :
            data = next(train_data_iterator)
        except StopIteration :
            train_data_iterator = iter(train_dataloader)
            data = next(train_data_iterator)

        optimizer.zero_grad()

        diag, cordi, rank = data["diag"], data["cordi"], data["rank"]
        diag = diag.float().to(device)
        cordi = cordi.float().to(device)
        rank = rank.long().to(device)
        logits = model(dlg=diag, crd=cordi)

        loss = loss_ce(logits, rank)
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, 1)
        acc += torch.sum(rank == preds).item() 

        if step > 0 and step % args.logging_steps == 0 :
            lr = scheduler.get_last_lr()[0]
            acc = acc / (args.batch_size * args.logging_steps)
            info = {"train/loss": loss.item(), "train/acc": acc, "train/learning_rate" : lr, "train/step" : step}
            print(info)
            acc = 0.0
        

    path = os.path.join(args.model_path, f"gAIa-final.pt")
    print("Saving Model : %s" %path)
    torch.save(model.state_dict(), path)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    # input options
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')
    parser.add_argument('--seed', type=int, 
        default=42, 
        help='random seed'
    )
    parser.add_argument('--in_file_trn_dialog', type=str,
        default='../data/ddata.wst.txt.2021.6.9',
        help='training dialog DB'
    )
    parser.add_argument('--in_file_fashion', type=str,
        default='../data/mdata.txt.2021.10.18',
        help='fashion item metadata'
    )
    parser.add_argument('--img_feat_dir', type=str,
        default='../data/img_feats',
        help='fashion image feature directory'
    )
    parser.add_argument('--model_path', type=str,
        default='./gAIa_model',
        help='path to save/read model'
    )
    parser.add_argument('--model_file', type=str,
        default=None,
        help='model file name'
    )
    parser.add_argument('--subWordEmb_path', type=str,
        default='../data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat',
        help='path of subword embedding'
    )
    parser.add_argument('--learning_rate', type=float,
        default=1e-5,
        help='learning rate'
    )
    parser.add_argument('--warmup_ratio', type=float,
        default=0.05,
        help='warmup ratio of total steps'
    )
    parser.add_argument('--weight_decay', type=float,
        default=1e-3,
        help='weight decay'
    )
    parser.add_argument('--num_layers', type=int,
        default=6,
        help='the number of hidden layers'
    )
    parser.add_argument('--d_model', type=int,
        default=512,
        help='d_model of encoder'
    )
    parser.add_argument('--hidden_size', type=int,
        default=2048,
        help='hidden size of encoder'
    )
    parser.add_argument('--num_head', type=int,
        default=8,
        help='the number of encoder head'
    )
    parser.add_argument('--batch_size', type=int,
        default=4,
        help='batch size for training'
    )
    parser.add_argument('--dropout_prob', type=float,
        default=0.1,
        help='dropout prob.'
    )
    parser.add_argument('--epochs', type=int,
        default=10,
        help='epochs to training'
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
    parser.add_argument('--lamda', type=int,
        default=1000,
        help='lamda of ewc'
    )
    parser.add_argument('--text_feat_size', type=int,
        default=512,
        help='size of text feature'
    )
    parser.add_argument('--eval_node', type=str,
        default='[6000,6000,6000,200][2000,2000]',
        help='nodes of evaluation network'
    )
    parser.add_argument('--augmentation_size', type=int,
        default=1,
        help='# of data augmentation'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )
    parser.add_argument('--logging_steps', type=int,
        default=100,
        help='logging steps'
    )

    args = parser.parse_args()
    train(args)