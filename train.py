
import os
import torch
import random
import argparse
import numpy as np

from torch.utils.data import DataLoader

from dataset.dataset import FashionHowDataset
from dataset.collator import PaddingCollator

from utils.encoder import Encoder
from utils.augmentation import DataAugmentation
from utils.preprocessor import DiagPreprocessor
from utils.loader import MetaLoader, DialogueLoader

from models.tokenizer import SubWordEmbReaderUtil

def train(args) :

    # -- Seed
    seed_everything(args.seed)

    # -- Subword Embedding
    print('\nInitializing subword embedding')
    swer = SubWordEmbReaderUtil(args.subWordEmb_path)

    # -- Meta Data
    print("\nLoading Meta Data...")
    meta_loader = MetaLoader(args.in_file_fashion, swer)
    img2id, id2img, img_similarity = meta_loader.get_dataset()

    # -- Dialogue Data
    print("\nLoading Dialogue Data...")
    train_diag_loader = DialogueLoader(args.in_file_trn_dialog)
    train_raw_dataset = train_diag_loader.get_train_dataset()

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
    encoder = Encoder(swer, img2id, num_cordi=4, mem_size=args.mem_size)
    train_encoded_dataset = encoder(train_dataset)

    # -- Data Collator & Loader
    data_collator = PaddingCollator()
    train_torch_dataset = FashionHowDataset(dataset=train_encoded_dataset)
    train_dataloader = DataLoader(train_torch_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )
    breakpoint()

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
        default='./data/task1.ddata.wst.txt',
        help='training dialog DB'
    )
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
    parser.add_argument('--learning_rate', type=float,
        default=1e-5,
        help='learning rate'
    )
    parser.add_argument('--dropout_prob', type=float,
        default=0.1,
        help='dropout prob.'
    )
    parser.add_argument('--batch_size', type=int,
        default=32,
        help='batch size for training'
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
    parser.add_argument('--augmentation_size', type=int,
        default=5,
        help='# of data augmentation'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )
    parser.add_argument('--use_cl', type=bool,
        default=True,
        help='enable continual learning'
    )

    args = parser.parse_args()
    train(args)
