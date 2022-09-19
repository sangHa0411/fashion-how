
import copy
import random
import numpy as np

class DiagPreprocessor :
    """
    추출된 대화와 cordi를 전처리해서 모든 데이터가 통일성을 가지도록 하는 클래스
    해당 대화와 추출된 codri를 에서 cordi를 늦게 추천한 순서대로 3개를 선정하도록 합니다.
    여기서 몇몇 경우에서는 기존에 추출된 cordi가 3개가 안되는 경우도 있는데 
    이는 기존의 cordi에서 하나를 임의로 선정해서 (dummy label 제외) 유사한 아이템으로 바꿈으로서 cordi를 3개로 맞추어줍니다.
    """
    def __init__(self, num_rank, num_cordi) :
        self.num_rank = num_rank
        self.num_cordi = num_cordi

    def __call__(self, raw_dataset, img2id, id2img, img_similarity) :
        dataset = []
        for i in range(len(raw_dataset)) :
            diag = raw_dataset[i]["diag"]
            cordi = raw_dataset[i]["cordi"]
            reward = raw_dataset[i]["reward"]

            if len(cordi) == 0 :
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
                    rand_id = np.random.randint(len(cordi_unique))
                    org = cordi_unique[rand_id]
                    targets = [k for k in range(self.num_cordi) if "NONE" not in org[k]]
                    target_id = random.sample(targets, 1)[0]
                    target_img = org[target_id]
                    
                    img_id = img2id[target_id][target_img]
                    img_sim = img_similarity[target_id][img_id]
                    
                    rank_args = np.argsort(img_sim)[::-1][1:]
                    select_id = np.random.randint(50)
                    select_arg = rank_args[select_id]
                    select_img = id2img[target_id][select_arg]

                    aug = copy.deepcopy(org)
                    aug[target_id] = select_img
                    aug_list.append(aug)
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


        