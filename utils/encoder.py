
import numpy as np
from tqdm import tqdm

class Encoder :

    def __init__(self, swer, img2id, num_cordi, mem_size) :
        self.swer = swer
        self.img2id = img2id
        self.num_cordi = num_cordi
        self.mem_size = mem_size 

    def __call__(self, dataset) :
        encoded_dataset = []
        for data in tqdm(dataset) :
            diag = data["diag"]
            vectors = [self.swer.get_sent_emb(sen) for sen in diag]
            if len(vectors) >= self.mem_size :
                vectors = vectors[:self.mem_size]
                diag_array = np.array(vectors)
            else :
                vectors = [np.zeros(self.swer.get_emb_size()) for i in range(self.mem_size - len(vectors))] + vectors
                diag_array = np.array(vectors)

            cordi = data["cordi"]
            cordi_list = []
            for i, c in enumerate(cordi) :
                imgs = [self.img2id[j][c[j]] for j in range(self.num_cordi)]
                cordi_list.append(imgs)
            cordi_array = np.array(cordi_list)
            outer_array, top_array, bottom_array, shoes_array = cordi_array[:,0], cordi_array[:,1], cordi_array[:,2], cordi_array[:,3]

            encoded_dataset.append(
                {
                    "diag" : diag_array, 
                    "cordi" : (outer_array, top_array, bottom_array, shoes_array),
                    "rank" : data["reward"],
                }
            )
        return encoded_dataset