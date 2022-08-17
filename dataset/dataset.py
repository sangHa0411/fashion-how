from torch.utils.data import Dataset

class FashionHowDataset(Dataset):
    def __init__(self, dataset):
        self.diag = dataset["diag"]
        self.cordi = dataset["cordi"]
        self.rank = dataset["rank"] if "rank" in dataset else None
            
    def __len__(self):
        return len(self.diag)

    def __getitem__(self, idx):
        diag = self.diag[idx]
        cordi = self.cordi[idx]

        if self.rank == None :
            return diag, cordi
        else :
            rank = self.rank[idx]
            return diag, cordi, rank
