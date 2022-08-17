
import numpy as np
from tqdm import tqdm

class Encoder :

    def __init__(self, swer, img2id, num_cordi, mem_size) :
        self.swer = swer
        self.img2id = img2id
        self.num_cordi = num_cordi
        self.mem_size = mem_size 

    def __call__(self, dataset) :
        diag_list, cordi_list, rank_list = [], [], []
        for data in tqdm(dataset) :
            diag = data["diag"]
            vectors = [self.swer.get_sent_emb(sen).tolist() for sen in diag]
            if len(vectors) >= self.mem_size :
                vectors = vectors[:self.mem_size]
            else :
                vectors = [np.zeros(self.swer.get_emb_size()).tolist() for i in range(self.mem_size - len(vectors))] + vectors

            cordi = data["cordi"]
            cordi_rows = []
            for c in cordi :
                imgs = [self.img2id[j][c[j]] for j in range(self.num_cordi)]
                cordi_rows.append(imgs)

            diag_list.append(vectors)
            cordi_list.append(cordi_rows)
            if "reward" in data :
                rank_list.append(data["reward"])

        if "reward" in dataset[0] :
            dataset = {"diag" : diag_list, "cordi" : cordi_list, "rank" : rank_list}
        else :
            dataset = {"diag" : diag_list, "cordi" : cordi_list}
        return dataset