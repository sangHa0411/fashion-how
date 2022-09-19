import copy
import random
import collections
import pandas as pd 
import numpy as np
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
    """
    mdata.wst.txt.2021.10.18 파일을 불러와서 분석 및 전처리하는 클래스
    """
    def __init__(self, path, swer) :
        self.path = path
        self.swer = swer

    def get_dataset(self,) :
        raw_dataset = self._load()
        img2id, id2img, img_similarity = self._get_mapping(raw_dataset)
        return img2id, id2img, img_similarity

    def _load(self) :
        """
        각각의 옷에 대한 설명을 하나로 모아서 각 옷을 key로 하고 해당 key에 대한 value를 문장 리스트를 구성해서 저장을 합니다.
        """
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
        """
        해당 옷에 카테고리별로 분류합니다. : img_category
        각 분류에 따라서 그 분류에 해당되는 옷들에 대해서 represenation vector를 구합니다. : img_vectors
        해당 옷에 대해서 그 카테고리에 있는 옷을 기준대로 representation vector들 간의 코사인 유사도를 구해서 그 결과를 저장합니다. : img_similarity
        카테고리의 갯수가 4개이고 해당 4개를 각각의 img 에서 id로, id에서 img로 mapping하는 dict 타입을 생성합니다. : img2id, id2img
        """
        img2description = {m["name"] : m["fashion_description"] for m in mdataset}
        img_list = list(img2description.keys())
        img_category = collections.defaultdict(list)

        for img in img_list :
            img_categorty = position_of_fashion_item(img)
            img_category[img_categorty].append(img)

        img_vectors = collections.defaultdict(dict) # img to vector
        for i in img_category :
            sub_img_list = img_category[i]

            for k in sub_img_list :
                desc = img2description[k]
                embed = [self.swer.get_sent_emb(d) for d in desc]
                vector = np.mean(embed, axis=0)
                img_vectors[i][k] = vector

        img_similarity = collections.defaultdict(list) # img to similarity

        for i in img_vectors :
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

        img2id[0]["NONE-OUTER"] = len(img2id[0])
        img2id[1]["NONE-TOP"] = len(img2id[1])
        img2id[2]["NONE-BOTTOM"] = len(img2id[2])
        img2id[3]["NONE-SHOES"] = len(img2id[3])

        id2img[0][len(id2img[0])] = "NONE-OUTER"
        id2img[1][len(id2img[1])] = "NONE-TOP"
        id2img[2][len(id2img[2])] = "NONE-BOTTOM"
        id2img[3][len(id2img[3])] = "NONE-SHOES"

        return img2id, id2img, img_similarity


class DialogueTrainLoader :
    """
    task1.ddata.wst.txt 와 같이 훈련 데이터를 불러와서 분석하고 전처리하는 클래스
    """
    def __init__(self, path,) :
        self.path = path

    def get_dataset(self, ) :
        df = self._load()
        stories = self._split(df)

        ddataset = []
        for i,d in enumerate(stories):
            d, c, r = self._extract(d)    
            data = {"diag" : d, "cordi" : c, "reward" : r}
            ddataset.append(data)
        return ddataset

    def _load(self, ) :
        """
        데이터를 불러와서 id, utterance, description, tag 등으로 구분해서 dataframe 형식으로 저장합니다.
        """
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
        """
        tag를 기반으로 해서 dataframe으로 일괄적으로 저장되어 있는 데이터를 대화 기준으로 구분합니다.
        """
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
        """
        코디가 제안한 조합 중에서는 몇몇 경우에서는 4가지가 아닌 2가지, 3가지가 제안되는 경우가 있는데 
        통일성을 위해서 해당 카테고리에 dummy label을 추가힙니다.
        """
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
        """
        해당 대화에서 대화, tag 등을 구분해서 추천된 cordi 등을 가져옵니다. 
        이 함수를 통해서는 코디가 제안한 횟수 만큼 cordi가 생성이 됩니다.
        각 코디별로 outer, top, bottom, shoes 순으로 정렬을 하게 되고 계 중에 없는 옷이 있다면 dummy label로 채웁니다.
        추후에 preprocessor.py에 있는 DiagPreprocessor을 통해서 위에서 생성된 코디 중에서 최종 3가지를 선정하게 됩니다.
        """
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

class DialogueTestLoader :
    """
    cl_eval_task1.wst.dev 와 같은 dev 혹은 test 데이터를 불러와서 분석하고 전처리하기 위한 클래스
    """
    def __init__(self, path, eval_flag) :
        self.path = path
        self.eval_flag = eval_flag

    def get_dataset(self, ) :
        ddata = self._load()
        stories = self._split(ddata)

        dataset = []
        for i in range(len(stories)):
            story = stories[i]
            if self.eval_flag :
                d, c, r = self._extract(story)
                data = {"diag" : d, "cordi" : c, "reward" : r}
            else :
                d, c = self._extract(story)
                data = {"diag" : d, "cordi" : c}
            dataset.append(data)
        return dataset

    def _load(self, ) :
        with open(self.path, encoding="euc-kr", mode="r") as f :
            ddata = f.readlines()
        return ddata

    def _split(self, dataset) :
        """
        해당 데이터를 ;를 기준으로 분류해서 대화로 나누는 함수입니다.
        """
        start_ids = []
        for i, r in enumerate(dataset) :
            if ";" in r :
                start_ids.append(i)

        stories = []
        for i in range(len(start_ids) - 1) :
            prev = start_ids[i]
            cur = start_ids[i+1]
            story = dataset[prev:cur]
            stories.append(story)
        return stories

    def _extract(self, story) :
        """
        분류된 대화 내에서 코디 부분과 대화 부분을 구분하는 함수입니다.
        """
        num = int(story[0][2:-1])
        desc = [row.split("\t")[1].strip() 
            for row in story[1:-3]
        ]
        cordi = [row.split("\t")[1].strip().split(" ") 
            for row in story[-3:]
        ]
        cordi = self._preprocess(cordi)

        if self.eval_flag :
            cordi, ranks = self._shuffle(cordi)
            return desc, cordi, ranks
        else :
            return desc, cordi

    def _preprocess(self, cordi) :
        """
        추천된 코디의 순서를 outer, top, bottom, shoes 순서대로 정렬하기 위한 함수입니다.
        그리고 그 과정 중에서 없는 것이 있다면 dummy label로 채우게 됩니다.
        """
        cordi = [
            [item.split("_")[-1] for item in c]
            for c in cordi
        ]
        cordi = [
            {position_of_fashion_item(item) : item for item in c} 
            for c in cordi
        ]
        cordi = [self._add_dummy(c) for c in cordi]
        return cordi

    def _shuffle(self, cordi) :
        """
        평가 데이터의 label은 항시 첫번째 추천 > 두 번째 추천 > 세 번째 추천이 되는데 
        이를 섞어주고 그에 맞게 label로 바꾸도록 했습니다.
        """
        ranks = [0, 1, 2]
        random.shuffle(ranks)

        cordi = [cordi[r] for r in ranks]
        return cordi, ranks

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