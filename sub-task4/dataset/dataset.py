import os
import numpy as np
from torch.utils.data import Dataset

class FashionHowDataset(Dataset):
    def __init__(self, dataset, img_feat_dir):
        self.diag = dataset["diag"]
        self.cordi = dataset["cordi"]
        self.rank = dataset["rank"] if "rank" in dataset else None
        self.img_feat_dir = img_feat_dir

    def __len__(self):
        return len(self.diag)

    def __getitem__(self, idx):
        diag = self.diag[idx]
        cordi = self.cordi[idx]

        cordi_array = []
        for c in cordi :
            vectors = []
            for cloth in c :
                if "NONE" in cloth :
                    vector = np.zeros((4, 2048)).tolist()
                else : 
                    file_name = os.path.join(self.img_feat_dir, cloth + "_feat.npy")
                    vector = np.load(file_name).tolist()
                vectors.append(vector)
            cordi_array.append(vectors)

        if self.rank == None :
            return diag, cordi_array
        else :
            rank = self.rank[idx]
            return diag, cordi_array, rank