
import copy
import random
import numpy as np
from tqdm import tqdm

class DataAugmentation :

    def __init__(self, num_aug) :
        self.num_aug = num_aug

    def __call__(self, dataset, img2id, id2img, img_similarity) :
        aug_dataset = []
        for d in tqdm(dataset) :
            aug_dataset.append(d)

            diag = d["diag"]
            cordi = d["cordi"]
            reward = d["reward"]

            for j in range(self.num_aug) :
                source_id = np.random.randint(3)
                source = cordi[source_id]

                targets = [k for k in range(4) if "NONE" not in source[k]]
                target_id = random.sample(targets, 1)[0]
                target_img = source[target_id]

                img_id = img2id[target_id][target_img]
                img_sim = img_similarity[target_id][img_id]
                
                rank_args = np.argsort(img_sim)[::-1][1:]
                select_id = np.random.randint(50)
                select_arg = rank_args[select_id]
                select_img = id2img[target_id][select_arg]

                aug = copy.deepcopy(source)
                aug[target_id] = select_img

                if source_id == 0 :
                    data = {"diag" : diag, 
                        "cordi" : [cordi[0]] + [cordi[np.random.randint(1,3)]] + [aug], 
                        "reward" : reward
                    }
                else :
                    data = {"diag" : diag, 
                        "cordi" : cordi[:source_id] + [aug] + cordi[source_id+1:], 
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