import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.loader import Loader
from utils.preprocessor import Preprocessor
from models.model import Model
from models.dataset import ImageDataset

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

    daily_pred_list = []
    gender_pred_list = []
    emb_pred_list = []

    for i in range(FOLD_SIZE) :
        print("\n%dth Model inference" %i)
        model = Model(HIDDEN_SIZE,
            label_size["daily"],
            label_size["gender"],
            label_size["embellishment"],
            DROPOUT_PROB,
            pretrained=False
        )
        model_path = f"/home/work/model/checkpoints/fold{i}/checkpoint-2500.pt"
        model.load_state_dict(torch.load(model_path, map_location=cuda_str))
        model.to(device)

        # -- Predict
        with torch.no_grad() :
            model.eval()

            daily_predictions = []
            gender_predictions = []
            emb_predictions = []

            for eval_data in dataloader :
                img = eval_data["image"].to(device)
                daily_logit, gender_logit, emb_logit = model(img)

                daily_pred = torch.argmax(daily_logit, -1)
                gender_pred = torch.argmax(gender_logit, -1)
                emb_pred = torch.argmax(emb_logit, -1)

                daily_predictions.extend(daily_logit.detach().cpu().numpy().tolist())
                gender_predictions.extend(gender_logit.detach().cpu().numpy().tolist())
                emb_predictions.extend(emb_logit.detach().cpu().numpy().tolist())

            daily_pred_list.append(daily_predictions)
            gender_pred_list.append(gender_predictions)
            emb_pred_list.append(emb_predictions)

    daily_id_list = []
    gender_id_list = []
    emb_id_list = []
    print("\nEnsemable Logits")
    for i in range(len(df)) :
        daily_preds = [daily_predictions[j][i] for j in range(FOLD_SIZE)]
        daily_pred_sum = np.sum(daily_preds, axis=0)
        daily_pred_sum.argmax()
        daily_id_list.append(daily_pred_sum)
            
        gender_preds = [gender_predictions[j][i] for j in range(FOLD_SIZE)]
        gender_pred_sum = np.sum(gender_preds, axis=0)
        gender_pred_sum.argmax()
        gender_id_list.append(gender_pred_sum)

        emb_preds = [emb_predictions[j][i] for j in range(FOLD_SIZE)]
        emb_pred_sum = np.sum(emb_preds, axis=0)
        emb_pred_sum.argmax()
        emb_id_list.append(emb_pred_sum)

    print("\nSaving results")
    df['Daily'] = daily_id_list.astype(int)
    df['Gender'] = gender_id_list.astype(int)
    df['Embellishment'] = emb_id_list.astype(int)
    df.to_csv('/home/work/model/prediction.csv', index=False)

if __name__ == '__main__':
    predict()

