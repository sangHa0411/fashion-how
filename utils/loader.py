import copy
import collections
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def position_of_fashion_item(item):
    prefix = item[0:2]
    if prefix=='JK' or prefix=='JP' or prefix=='CT' or prefix=='CD' or prefix=='VT':
        idx = 0
    elif prefix=='KN' or prefix=='SW' or prefix=='SH' or prefix=='BL' :
        idx = 1
    elif prefix=='SK' or prefix=='PT' or prefix=='OP' :
        idx = 2
    elif prefix=='SE':
        idx = 3
    else:
        raise ValueError('{} do not exists.'.format(item))
    return idx

class MetaLoader :

    def __init__(self, path, swer) :
        self.path = path
        self.swer = swer

    def get_dataset(self,) :
        raw_dataset = self._load()
        img2id, id2img, img_similarity = self._get_mapping(raw_dataset)
        return img2id, id2img, img_similarity

    def _load(self) :
        with open(self.path, encoding="euc-kr", mode="r") as f :
            mdata = f.readlines()

        mdataset = []
        prev_name = ""
        for i in range(len(mdata)) :
            row = mdata[i].split("\t")
            row = [r.strip() for r in row]
            if row[0] != prev_name :
                data = {"name" : row[0], "category" : row[1], "fashion_type" : row[2], "fashion_characteristic" : [row[3]], "fashion_description" : [row[4]]}
                prev_name = row[0]
                mdataset.append(data)
            else :
                data["fashion_characteristic"].append(row[3])
                data["fashion_description"].append(row[4])
        
        return mdataset

    def _get_mapping(self, mdataset) :
        img2description = {m["name"] : m["fashion_description"] for m in mdataset}
        img_list = list(img2description.keys())
        img_category = collections.defaultdict(list)

        for img in img_list :
            img_categorty = position_of_fashion_item(img)
            img_category[img_categorty].append(img)

        img_vectors = collections.defaultdict(dict) # img to vector
        for i in tqdm(img_category) :
            sub_img_list = img_category[i]

            for k in sub_img_list :
                desc = img2description[k]
                embed = [self.swer.get_sent_emb(d) for d in desc]
                vector = np.mean(embed, axis=0)
                img_vectors[i][k] = vector

        img_similarity = collections.defaultdict(list) # img to similarity

        for i in tqdm(img_vectors) :
            vectors = list(img_vectors[i].values())
            array = np.array(vectors)
            for j in range(len(vectors)) :
                org_vector = vectors[j]
                org_vector = np.expand_dims(org_vector, axis=0)
                similarity = cosine_similarity(org_vector, array)
                img_similarity[i].append(similarity[0])

        img2id = collections.defaultdict(dict) # img to id
        id2img = collections.defaultdict(dict) # id to img
        for i in img_category :
            mapping = img_category[i]
            for j,k in enumerate(mapping) :
                img2id[i][k] = j
                id2img[i][j] = k

        img2id[0][len(img2id[0])] = "NONE-OUTER"
        img2id[1][len(img2id[1])] = "NONE-TOP"
        img2id[2][len(img2id[2])] = "NONE-BOTTOM"
        img2id[3][len(img2id[3])] = "NONE-SHOES"

        id2img[0]["NONE-OUTER"] = len(id2img[0])
        id2img[1]["NONE-TOP"] = len(id2img[1])
        id2img[2]["NONE-BOTTOM"] = len(id2img[2])
        id2img[3]["NONE-SHOES"] = len(id2img[3])

        return img2id, id2img, img_similarity


class DialogueLoader :

    def __init__(self, path,) :
        self.path = path

    def get_dataset(self, ) :
        df = self._load()
        stories = self._split(df)

        ddataset = []
        for i,d in enumerate(tqdm(stories)):
            d, c, r = self._extract(d)    
            data = {"diag" : d, "cordi" : c, "reward" : r}
            ddataset.append(data)
        return ddataset

    def _load(self, ) :
        with open(self.path, encoding="euc-kr", mode="r") as f :
            ddata = f.readlines()

        id_list = []
        utter_list = []
        desc_list = []
        tag_list = []

        for i in range(len(ddata)) :

            row = ddata[i].split("\t")
            row = [r.strip() for r in row]

            id_list.append(row[0])
            utter_list.append(row[1])
            desc_list.append(row[2])
            if len(row) == 4 :
                tag_list.append(row[3])
            else :
                tag_list.append("")

        df = pd.DataFrame({"id" : id_list, "utterance" : utter_list, "description" : desc_list, "tag" : tag_list})
        return df

    def _split(self, df) :
        start_ids = []
        for i in range(len(df)) :
            tag = df.iloc[i]["tag"]
            if tag == "INTRO" :
                start_ids.append(i)

        stories = []
        for i in range(len(start_ids) - 1) :
            prev = start_ids[i]
            cur = start_ids[i+1]
            sub_df = df[prev:cur]
            stories.append(sub_df)

        stories.append(df[start_ids[i+1]:])
        return stories

    def _add_dummy(self, item) :
        if 0 not in item :
            item[0] = "NONE-OUTER"
        
        if 1 not in item :
            item[1] = "NONE-TOP"
        
        if 2 not in item :
            item[2] = "NONE-BOTTOM"
        
        if 3 not in item :
            item[3] = "NONE-SHOES"
        return item

    def _extract(self, stories) :
        descriptions = [stories.iloc[i]["description"] for i in range(len(stories)) 
            if stories.iloc[i]["utterance"] != "<AC>"
        ]

        coordi = []
        reward = []

        i = 0
        clothes = None
        while i < len(stories) :
            row = stories.iloc[i]
            tag = row["tag"]
            utter = row["utterance"]
            
            if "USER" in tag :
                coordi.append(clothes)
                reward.append(tag)

            if utter == "<AC>" :
                desc = row["description"]

                if clothes == None :
                    clothes = {position_of_fashion_item(c) : c for c in desc.split(" ")}
                    clothes = self._add_dummy(clothes)
                else :
                    clothes_temp = copy.deepcopy(clothes)
                    for c in desc.split(" ") :
                        c_id = position_of_fashion_item(c)
                        clothes_temp[c_id] = c
                    clothes = clothes_temp
            i += 1

        return descriptions, coordi, reward