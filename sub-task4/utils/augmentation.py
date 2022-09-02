
import copy
import random
import numpy as np
from tqdm import tqdm

class DataAugmentation :

    def __init__(self, num_rank, num_cordi, num_aug) :
        self.num_rank = num_rank
        self.num_cordi = num_cordi
        self.num_aug = num_aug

    def __call__(self, dataset, img2id, id2img, img_similarity) :
        aug_dataset = []
        for d in tqdm(dataset) :
            aug_dataset.append(d)

            diag = d["diag"]
            cordi = d["cordi"]
            reward = d["reward"]

            for j in range(self.num_aug) :

                org_cordi = cordi[0]
                targets = [k for k in range(self.num_cordi) if "NONE" not in org_cordi[k]]

                aug_cordi = copy.deepcopy(org_cordi)
                target_ids = random.sample(targets, 2)
                for t_id in target_ids :
                    t_img = org_cordi[t_id]

                    img_id = img2id[t_id][t_img]
                    img_sim = img_similarity[t_id][img_id]
                    
                    rank_args = np.argsort(img_sim)[::-1][1:]
                    select_id = np.random.randint(100)
                    select_arg = rank_args[select_id]
                    select_img = id2img[t_id][select_arg]

                    aug_cordi[t_id] = select_img

                data = {"diag" : diag, 
                    "cordi" : [cordi[0], cordi[np.random.randint(1,3)]] + [aug_cordi],
                    "reward" : reward
                }
                aug_dataset.append(data)

        shuffled_dataset = []
        for data in aug_dataset :
            shuffled_dataset.append(self._shuffle(data))
        return shuffled_dataset

    def _shuffle(self, data) :
        diag = data["diag"]
        cordi = data["cordi"]
        reward = data["reward"]

        ranks = [0, 1, 2]
        random.shuffle(ranks)

        cordi = [cordi[r] for r in ranks]
        reward = ranks.index(0)

        shuffled = {"diag" : diag, "cordi" : cordi, "reward" : reward}
        return shuffled