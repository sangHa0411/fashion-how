import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.loader import Loader
from utils.preprocessor import Preprocessor
from models.model import Model
from models.dataset import ImageDataset
from tqdm import tqdm

FOLD_SIZE = 5
IMG_SIZE = 224
HIDDEN_SIZE = 2048
DROPOUT_PROB = 0.1
BATCH_SIZE = 16
INFO_PATH = "/home/work/data/task1/info_etri20_emotion_test.csv"
IMAGE_DIR = "/home/work/data/task1/test/"
NUM_WORKERS = 2

def predict():

    # -- Device
    cuda_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(cuda_str)

    # -- Loading Data
    print("\nLoading Dataset")
    df = pd.read_csv(INFO_PATH)
    loader = Loader(INFO_PATH, IMAGE_DIR, IMG_SIZE)
    dataset = loader.get_dataset()

    # -- Preprocessing Data
    print("\nPreprocessing Dataset")
    preprocessor = Preprocessor(IMG_SIZE)
    dataset = preprocessor(dataset)

    # -- Torch Dataset
    print("\nPreparing Dataset")
    dataset = ImageDataset(dataset)
    dataloader = DataLoader(dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )

    # -- model    
    label_size = {"daily" : 7, "gender" : 6, "embellishment" : 3}


    print("\nLoading Model")
    model = Model(HIDDEN_SIZE,
        label_size["daily"],
        label_size["gender"],
        label_size["embellishment"],
        DROPOUT_PROB,
        pretrained=False
    )
    model_path = f"/home/work/model/checkpoints/model1/checkpoint-1000.pt"
    model.load_state_dict(torch.load(model_path, map_location=cuda_str))
    model.to(device)

    daily_predictions = []
    gender_predictions = []
    emb_predictions = []

    # -- Predict
    with torch.no_grad() :
        model.eval()

        for eval_data in tqdm(dataloader) :
            img = eval_data["image"].to(device)
            daily_logit, gender_logit, emb_logit = model(img)

            daily_pred = torch.argmax(daily_logit, -1)
            gender_pred = torch.argmax(gender_logit, -1)
            emb_pred = torch.argmax(emb_logit, -1)

            daily_predictions.extend(daily_pred.detach().cpu().numpy().tolist())
            gender_predictions.extend(gender_pred.detach().cpu().numpy().tolist())
            emb_predictions.extend(emb_pred.detach().cpu().numpy().tolist())

    daily_predictions = np.array(daily_predictions)
    gender_predictions = np.array(gender_predictions)
    emb_predictions = np.array(emb_predictions)

    print("\nSaving results")
    df['Daily'] = daily_predictions.astype(int)
    df['Gender'] = gender_predictions.astype(int)
    df['Embellishment'] = emb_predictions.astype(int)
    df.to_csv('/home/work/model/prediction.csv', index=False)

if __name__ == '__main__':
    predict()

