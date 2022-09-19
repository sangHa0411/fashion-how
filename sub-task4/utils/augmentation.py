
import copy
import random
import numpy as np
from tqdm import tqdm

class DataAugmentation :
    """
    Preprocessor를 거친 대화, cordi 데이터를 대상으로 해당 cordi 중 임의로 하나를 선택합니다.
    해당 cordi에서 4가지 아이템 중에 두 개를 임의로 선택합니다. (dummy label 제외)
    선택된 아이템과 코사인 유사도 기반으로 유사한 아이템을 선택해서 추천을 하나 만들어 줌으로써 데이터를 추가합니다.
    """
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