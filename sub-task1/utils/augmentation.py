import copy
import numpy as np
from skimage import transform
from tqdm import tqdm

class CutMix :

    def __init__(self, num_aug) :
        self._num_aug = num_aug

    def __call__(self, dataset) :
        aug_dataset = []

        for i, data in enumerate(tqdm(dataset)) :
            # source
            org_img = data["image"]
            org_h, org_w, _ = org_img.shape
            org_daily = data["daily"]
            org_gender = data["gender"]
            org_emb = data["embellishment"]

            aug_dataset.append(data)

            rand_id_list = []
            for j in range(self._num_aug) :
                rand_id = np.random.randint(len(dataset))
                while i == rand_id or i in rand_id_list :
                    rand_id = np.random.randint(len(dataset))
                rand_id_list.append(rand_id)

                # target
                tar_img = copy.deepcopy(dataset[rand_id]["image"])
                
                tar_daily = dataset[rand_id]["daily"]
                tar_gender = dataset[rand_id]["gender"]
                tar_emb = dataset[rand_id]["embellishment"]

                lam = np.random.uniform(0, 1)
                val = np.math.sqrt(1 - lam)
                r_y = np.random.randint(0, org_h)
                r_h = val * org_h
                
                r_x = np.random.randint(0, org_w)
                r_w = val * org_w

                # width
                x1 = round(r_x - (r_w / 2))
                x1 = x1 if x1 > 0 else 0
                x2 = round(r_x + (r_w / 2))
                x2 = x2 if x2 < org_w else org_w
                
                # height
                y1 = round(r_y - (r_h / 2))
                y1 = y1 if y1 > 0 else 0
                y2 = round(r_y + (r_y / 2))
                y2 = y2 if y2 < org_h else org_h

                new_img = copy.deepcopy(org_img)
                new_img[y1:y2, x1:x2, :] = tar_img[y1:y2, x1:x2, :]
                
                co = 1 - ((x2-x1) * (y2-y1) / (org_h * org_w))

                new_daily = co * org_daily + (1-co) * tar_daily
                new_gender = co * org_gender + (1-co) * tar_gender
                new_emb = co * org_emb + (1-co) * tar_emb

                new_data = {"image" : new_img, "daily" : new_daily, "gender" : new_gender, "embellishment" : new_emb}
                aug_dataset.append(new_data)

        return aug_dataset