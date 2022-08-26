
from itertools import accumulate
import os
import random
import argparse
import numpy as np
import torch
import wandb
from torch import optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from utils.loader import Loader
from utils.preprocessor import Preprocessor
from utils.augmentation import CutMix
from models.model import Model
from models.loss import loss_fn, acc_fn
from models.dataset import ImageDataset
from tqdm import tqdm

def train(args):

    # -- Seed
    seed_everything(args.seed)

    # -- Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Loading Data
    loader = Loader(args.info_path, args.image_dir, args.img_size)
    dataset = loader.get_dataset()
    print("The number of dataset : %d" %len(dataset))
    random.shuffle(dataset)

    for i in range(args.k_fold) :
        print("\nTraining %dth fold" %i)
        size = int(len(dataset) / args.k_fold)
        train_dataset = dataset[:i*size] + dataset[(i+1)*size:]
        eval_dataset = dataset[i*size:(i+1)*size]

        # # -- Data Augmentation
        # augmentation = CutMix(args.num_aug)
        # train_dataset = augmentation(train_dataset)

        # -- Preprocess Data
        preprocessor = Preprocessor(args.img_size)
        train_dataset = preprocessor(train_dataset)
        eval_dataset = preprocessor(eval_dataset)

        print("\nThe number of train dataset : %d" %len(train_dataset))
        print("\nThe number of eval dataset : %d" %len(eval_dataset))

        # -- Torch Dataset & Dataloader
        train_dataset = ImageDataset(train_dataset)
        train_dataloader = DataLoader(train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers
        )

        eval_dataset = ImageDataset(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )

        # -- model
        label_size = loader.get_label_size()
        model = Model(args.hidden_size,
            label_size["daily"],
            label_size["gender"],
            label_size["embellishment"],
            args.dropout_prob,
            pretrained=True
        )
        model.to(device)

        # -- Optimizer & Scheduler
        iteration = int(args.epochs / 2)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iteration, eta_min=0)

        # -- Wandb
        load_dotenv(dotenv_path="wandb.env")
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        name = f"FOLD:{i}_EP:{args.epochs}_BS:{args.batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}"
        wandb.init(
            entity="sangha0411",
            project="fashion-how",
            group=f"sub-task1",
            name=name
        )

        training_args = {"epochs": args.epochs, 
            "batch_size": args.batch_size, 
            "learning_rate": args.learning_rate, 
            "weight_decay": args.weight_decay, 
        }
        wandb.config.update(training_args)

        # -- Training
        print("\nTraining")
        train_data_iterator = iter(train_dataloader)
        total_steps = len(train_dataloader) * args.epochs
        for step in tqdm(range(total_steps)) :
            try :
                data = next(train_data_iterator)
            except StopIteration :
                train_data_iterator = iter(train_dataloader)
                data = next(train_data_iterator)

            optimizer.zero_grad()

            img = data["image"].to(device)
            daily = data["daily"].to(device)
            gender = data["gender"].to(device)
            emb = data["embellishment"].to(device)

            daily_logit, gender_logit, emb_logit = model(img)

            daily_loss = loss_fn(daily_logit, daily)
            gender_loss = loss_fn(gender_logit, gender)
            emb_loss = loss_fn(emb_logit, emb)

            loss = daily_loss + gender_loss + emb_loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            if step > 0 and step % args.logging_steps == 0 :
                info = {"train/learning_rate" : scheduler.get_last_lr()[0], 
                    "train/daily_loss": daily_loss.item(), 
                    "train/gender_loss" : gender_loss.item(), 
                    "train/embellishment_loss": emb_loss.item(),
                    "train/step" : step,
                }
                wandb.log(info)
                print(info)

            if step > 0 and step % args.save_steps == 0 :
                print("\nEvaluating model at %d step" %step)
                with torch.no_grad() :
                    model.eval()
                    daily_acc, gender_acc, emb_acc = 0.0, 0.0, 0.0

                    for eval_data in tqdm(eval_dataloader) :

                        img = eval_data["image"].to(device)
                        daily = eval_data["daily"].to(device)
                        gender = eval_data["gender"].to(device)
                        emb = eval_data["embellishment"].to(device)

                        daily_logit, gender_logit, emb_logit = model(img)

                        daily_acc += acc_fn(daily_logit, daily)
                        gender_acc += acc_fn(gender_logit, gender)
                        emb_acc += acc_fn(emb_logit, emb)

                    daily_acc /= len(eval_dataset)
                    gender_acc /= len(eval_dataset)
                    emb_acc /= len(eval_dataset)
                        
                    info = {"eval/daily_acc": daily_acc, 
                        "eval/gender_acc" : gender_acc, 
                        "eval/embellishment_acc": emb_acc,
                        "eval/step" : step,
                    }
                    wandb.log(info)  
                    print(info)
                    model.train()

                # path = os.path.join(args.save_path, f"fold{i}", f"checkpoint-{step}.pt")
                # torch.save(model.state_dict(), path)

        wandb.finish()


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
    parser.add_argument('--k-fold', type=int, 
        default=5, 
        help='k-fold'
    )
    parser.add_argument('--img_size', type=int, 
        default=224, 
        help='image size'
    )
    parser.add_argument('--hidden_size', type=int, 
        default=2048, 
        help='hidden size of model'
    )
    parser.add_argument('--num_aug', type=int, 
        default=5, 
        help='augmentation size'
    )
    parser.add_argument('--image_dir', type=str,
        default='/home/wkrtkd911/project/fashion-how/sub-task1/data/train',
        help='fahsion image directory'
    )
    parser.add_argument('--info_path', type=str,
        default='/home/wkrtkd911/project/fashion-how/sub-task1/data/info_etri20_emotion_train.csv',
        help='fashion image information path'
    )
    parser.add_argument('--save_path', type=str,
        default='checkpoints',
        help='path to save model'
    )
    parser.add_argument('--learning_rate', type=float,
        default=3e-5,
        help='learning rate'
    )
    parser.add_argument('--weight_decay', type=float,
        default=1e-3,
        help='weight decay'
    )
    parser.add_argument('--dropout_prob', type=float,
        default=0.1,
        help='dropout prob.'
    )
    parser.add_argument('--batch_size', type=int,
        default=128,
        help='batch size for training'
    )
    parser.add_argument('--eval_batch_size', type=int,
        default=32,
        help='batch size for evaluation'
    )
    parser.add_argument('--epochs', type=int,
        default=100,
        help='epochs to training'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )
    parser.add_argument('--logging_steps', type=int,
        default=50,
        help='logging steps'
    )
    parser.add_argument('--save_steps', type=int,
        default=100,
        help='save steps'
    )

    args = parser.parse_args()
    train(args)

