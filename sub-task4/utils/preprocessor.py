
import copy
import random
import numpy as np
from tqdm import tqdm

class DiagPreprocessor :

    def __init__(self, num_rank, num_cordi) :
        self.num_rank = num_rank
        self.num_cordi = num_cordi

    def __call__(self, raw_dataset, img2id, id2img, img_similarity) :
        dataset = []
        for i in tqdm(range(len(raw_dataset))) :
            diag = raw_dataset[i]["diag"]
            cordi = raw_dataset[i]["cordi"]
            reward = raw_dataset[i]["reward"]

            if "USER_SUCCESS" not in reward :
                continue

            valid_items = [k for k in range(self.num_cordi) if "NONE" not in cordi[-1][k]]
            if len(valid_items) < 2 :
                continue

            cordi_unique = []
            reward_unique = []
            j = len(cordi) - 1
            prev_cordi = None
            while j >= 0 :
                if prev_cordi == None :
                    cordi_unique.append(cordi[j])
                    reward_unique.append(reward[j])
                    prev_cordi = cordi[j]
                else :
                    if not self._equal(prev_cordi, cordi[j]) :
                        cordi_unique.append(cordi[j])
                        reward_unique.append(reward[j])
                        prev_cordi = cordi[j]
                j -= 1


            if len(cordi_unique) > self.num_rank :
                data = {"diag" : diag, "cordi" : cordi_unique[:3], "reward" : 0}    
            else :
                aug_size = self.num_rank - len(cordi_unique)
                aug_list = []
                for a in range(aug_size) :
                    org_cordi = cordi_unique[0]
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

                    aug_list.append(aug_cordi)
                data = {"diag" : diag, "cordi" : cordi_unique + aug_list, "reward" : 0}    
            dataset.append(data)

        return dataset

    def _equal(self, a, b) :
        for k, a_v in a.items() :
            if k not in b :
                return False
                
            b_v = b[k]
            if a_v != b_v :
                return False 
        return True


        