import os
import numpy as np
from tqdm import tqdm

class Encoder :
    """
    대화를 sstm_v4p49_np_n36134_d128.dat 파일을 기반으로 한 tokenizer를 통해서 벡터로 변환을 하고
    코디 추천은 인덱스 형태로 변환하면서 딕셔너리에서 numpy array로 변환됩니다.
    """
    def __init__(self, swer, num_cordi, mem_size) :
        self.swer = swer
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
            cordi_names = [
                [c[j] for j in range(self.num_cordi)]
                for c in cordi
            ]

            diag_list.append(vectors)
            cordi_list.append(cordi_names)
            if "reward" in data :
                rank_list.append(data["reward"])

        if "reward" in dataset[0] :
            dataset = {"diag" : diag_list, "cordi" : cordi_list, "rank" : rank_list}
        else :
            dataset = {"diag" : diag_list, "cordi" : cordi_list}
        return dataset