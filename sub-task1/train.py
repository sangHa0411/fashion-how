import torch
import random
import argparse
import importlib
import numpy as np
from trainer import Trainer
from torch.utils.data import DataLoader
from utils.loader import Loader
from utils.preprocessor import Preprocessor
from utils.augmentation import CutMix
from models.dataset import ImageDataset

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

    if args.do_eval :
        size = int(len(dataset) / 4)
        train_dataset = dataset[size:]
        eval_dataset = dataset[:size]
    else :
        train_dataset = dataset

    # -- Data Augmentation 
    print("\nAugment dataset using cutmix")
    augmentation = CutMix(args.num_aug)
    train_dataset = augmentation(train_dataset)
    print("The number of dataset: %d" %len(train_dataset))

    # -- Preprocess Data
    print("\nPreprocessing Dataset")
    preprocessor = Preprocessor(args.img_size)
    train_dataset = preprocessor(train_dataset)
    if args.do_eval :
        eval_dataset = preprocessor(eval_dataset)
        
    print("\nThe number of train dataset : %d" %len(train_dataset))
    if args.do_eval :
        print("The number of eval dataset : %d\n" %len(eval_dataset))

    # -- Torch Dataset & Dataloader
    train_dataset = ImageDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )

    if args.do_eval :
        eval_dataset = ImageDataset(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
    else :
        eval_dataloader = None

    # -- model
    label_size = loader.get_label_size()

    model_name = args.backbone 
    model_lib = importlib.import_module("models.model")
    model_class = getattr(model_lib, model_name)
    model = model_class(args.hidden_size,
        label_size["daily"],
        label_size["gender"],
        label_size["embellishment"],
        args.dropout_prob,
        pretrained=True
    )
    model.to(device)

    # -- Training
    trainer = Trainer(args, device, model, train_dataloader, eval_dataloader)
    trainer.train()


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
    parser.add_argument('--do_eval', type=bool, 
        default=False, 
        help='evaluation flag'
    )
    parser.add_argument('--loss', type=str, 
        default="softmax", 
        help='loss function'
    )
    parser.add_argument('--num_model', type=int, 
        default=0, 
        help='number of model'
    )
    parser.add_argument('--img_size', type=int, 
        default=224, 
        help='orignal image size'
    )
    parser.add_argument('--hidden_size', type=int, 
        default=2048, 
        help='hidden size of model'
    )
    parser.add_argument('--backbone', type=str, 
        default="resnet", 
        help='image classification model backbone'
    )
    parser.add_argument('--warmup_ratio', type=float, 
        default=0.05, 
        help='warmup ratio of total steps'
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
        default=5e-5,
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
        default=5,
        help='epochs to training'
    )
    parser.add_argument('--max_steps', type=int, 
        default=-1, 
        help='max steps of training'
    )
    parser.add_argument('--num_workers', type=int,
        default=4,
        help='the number of workers for data loader'
    )
    parser.add_argument('--logging_steps', type=int,
        default=100,
        help='logging steps'
    )
    parser.add_argument('--save_steps', type=int,
        default=500,
        help='save steps'
    )

    args = parser.parse_args()
    train(args)
