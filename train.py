
import os
import torch
import random
import argparse
import wandb
import numpy as np
from tqdm import tqdm
from scipy import stats
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from dataset.dataset import FashionHowDataset
from dataset.collator import PaddingCollator

from utils.encoder import Encoder
from utils.augmentation import DataAugmentation
from utils.preprocessor import DiagPreprocessor
from utils.loader import MetaLoader, DialogueTrainLoader, DialogueTestLoader

from models.model import Model
from models.tokenizer import SubWordEmbReaderUtil

def train(args) :

    # -- Seed
    seed_everything(args.seed)

    # -- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    eval_diag_loader = DialogueTestLoader(args.in_file_tst_dialog, eval_flag=True)
    eval_dataset = eval_diag_loader.get_dataset()
    eval_rewards = [eval_data.pop("reward")for eval_data in eval_dataset]

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
    eval_encoded_dataset = encoder(eval_dataset)

    # -- Data Collator & Loader
    data_collator = PaddingCollator()
    train_torch_dataset = FashionHowDataset(dataset=train_encoded_dataset)
    train_dataloader = DataLoader(train_torch_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )

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
        img_feat_size=args.img_feat_size
    )

    # -- Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss_ce = torch.nn.CrossEntropyLoss().to(device)

    # -- Wandb
    load_dotenv(dotenv_path="wandb.env")
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    name = f"EP:{args.epochs}_BS:{args.batch_size}_LR:{args.learning_rate}"
    wandb.init(
        entity="sangha0411",
        project="fashion-how",
        group="baseline",
        name=name
    )

    training_args = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate}
    wandb.config.update(training_args)

    # -- Training
    acc = 0.0
    model.to(device)
    train_data_iterator = iter(train_dataloader)
    total_steps = len(train_dataloader) * args.epochs
    for step in tqdm(range(total_steps)) :
        try :
            data = next(train_data_iterator)
        except StopIteration :
            train_data_iterator = iter(train_dataloader)
            data = next(train_data_iterator)

        optimizer.zero_grad()

        diag, cordi, rank = data["diag"], data["cordi"], data["rank"]
        diag = diag.float().to(device)
        cordi = cordi.long().to(device)
        rank = rank.long().to(device)
        logits = model(dlg=diag, crd=cordi)

        loss = loss_ce(logits, rank)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, 1)
        acc += torch.sum(rank == preds).item()

        if step > 0 and step % args.logging_steps == 0 :
            info = {"loss": loss.item(), "acc": acc / args.logging_steps}
            print(info)
            wandb.log(info)
            acc = 0.0

        if step > 0 and step % args.eval_steps == 0 :
            eval_predictions = []
            with torch.no_grad() :
                model.eval()
                for eval_data in eval_dataloader :
                    diag, cordi = eval_data["diag"], eval_data["cordi"]
                    diag = diag.float().to(device)
                    cordi = cordi.long().to(device)
                    rank = rank.long().to(device)
                    logits = model(dlg=diag, crd=cordi)

                    preds = torch.argsort(logits, -1).detach().cpu().numpy()
                    eval_predictions.extend([pred.tolist()[::-1] for pred in preds])
            
            eval_tau = 0.0
            for i in range(len(eval_rewards)) :
                tau, _ = stats.weightedtau(eval_rewards[i], eval_predictions[i])
                eval_tau += tau

            eval_tau /= len(eval_rewards)
            wandb.log({"eval/tau" : eval_tau})
            print("Evaluation Tau : %.4f" %eval_tau)
            model.train()

            path = os.path.join(args.model_path, f"checkpoint-{step}.pt")
            # torch.save(model.state_dict(), path)

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
        default=8,
        help='batch size for training'
    )
    parser.add_argument('--eval_batch_size', type=int,
        default=16,
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
    parser.add_argument('--img_feat_size', type=int,
        default=512,
        help='size of image feature'
    )
    parser.add_argument('--eval_node', type=str,
        default='[6000,6000,6000,200][2000,2000]',
        help='nodes of evaluation network'
    )
    parser.add_argument('--augmentation_size', type=int,
        default=5,
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
    parser.add_argument('--eval_steps', type=int,
        default=500,
        help='eval steps'
    )
    parser.add_argument('--use_cl', type=bool,
        default=True,
        help='enable continual learning'
    )

    args = parser.parse_args()
    train(args)
