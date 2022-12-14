
import os
import torch
import importlib
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.loader import Loader
from utils.preprocessor import Preprocessor
from models.model import ResNetFeedForwardModel
from models.dataset import ImageDataset

NUM_MODEL = 4
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
    daily_pred_list, gender_pred_list, emb_pred_list = [], [], []

    # 총 4개의 모델을 앙상블합니다.
    # 모델 경로들
    model_paths = [
        "/home/work/model/checkpoints/model1/checkpoint-2400.pt",
        "/home/work/model/checkpoints/model2/checkpoint-2400.pt",
        "/home/work/model/checkpoints/model3/checkpoint-2400.pt",
        "/home/work/model/checkpoints/model4/checkpoint-2400.pt"
    ]
    # 모델 class 이름들
    model_names = [
        "ResNetFeedForwardModel",
        "DenseNetFeedForwardModel",
        "ResNetResidualConnectionModel",
        "DenseNetResidualConnectionModel",
    ]

    model_lib = importlib.import_module("models.model")

    # 순차적으로 모델을 불러와서 추론을 진행, 추론 결과를 logit 형태로 저장합니다.
    for i in range(NUM_MODEL) :
        print("\nLoading %dth Model" %(i+1))
        model_path = model_paths[i]
        model_name = model_names[i]
        model_class = getattr(model_lib, model_name)
        model = model_class(HIDDEN_SIZE,
            label_size["daily"],
            label_size["gender"],
            label_size["embellishment"],
            DROPOUT_PROB,
            pretrained=False
        )
        model.load_state_dict(torch.load(model_path, map_location=cuda_str))
        model.to(device)

        daily_predictions, gender_predictions, emb_predictions = [], [], []
        # -- Predict
        with torch.no_grad() :
            model.eval()
            for eval_data in tqdm(dataloader) :
                img = eval_data["image"].to(device)
                daily_logit, gender_logit, emb_logit = model(img)

                daily_predictions.extend(daily_logit.detach().cpu().numpy().tolist())
                gender_predictions.extend(gender_logit.detach().cpu().numpy().tolist())
                emb_predictions.extend(emb_logit.detach().cpu().numpy().tolist())

        daily_pred_list.append(daily_predictions)
        gender_pred_list.append(gender_predictions)
        emb_pred_list.append(emb_predictions)

    # -- Ensembale
    print("\nEnsemable predictions")
    # 저장된 logit 값들을 기반으로 soft voting 형식으로 앙상블을 진행합니다.
    daily_id_list, gender_id_list, emb_id_list = [], [], []
    for i in tqdm(range(len(df))) :
        daily_pred = np.sum([daily_pred_list[j][i] for j in range(NUM_MODEL)], axis=0)
        daily_id_list.append(daily_pred.argmax())

        gender_pred = np.sum([gender_pred_list[j][i] for j in range(NUM_MODEL)], axis=0)
        gender_id_list.append(gender_pred.argmax())

        emb_pred = np.sum([emb_pred_list[j][i] for j in range(NUM_MODEL)], axis=0)
        emb_id_list.append(emb_pred.argmax())

    print("\nSaving results")
    df['Daily'] = np.array(daily_id_list).astype(int)
    df['Gender'] = np.array(gender_id_list).astype(int)
    df['Embellishment'] = np.array(emb_id_list).astype(int)
    df.to_csv('/home/work/model/prediction.csv', index=False)

if __name__ == '__main__':
    predict()