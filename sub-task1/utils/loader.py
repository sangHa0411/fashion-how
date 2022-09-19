
import os
import copy
import numpy as np
import pandas as pd
import multiprocessing
import parmap
from tqdm import tqdm
from skimage import io, transform, color

class Loader :
    """
    데이터가 있는 디렉토리 경로를 받아서 거기에 있는 이미지를 불러오는 역할을 합니다.
    """
    def __init__(self, info_path, dir_path, img_size) :
        self._info_path = info_path
        self._dir_path = dir_path
        self._img_size = img_size
        self._df = pd.read_csv(info_path)

    def get_data(self, i) :
        """
        해당 인덱스의 이미지 이름을 통해서 실제 이미지를 skimage 라이브러리를 통해 불러옵니다.
        해당 이미지 사이즈의 비율을 파악해서 높이 너비 둘 중 긴 길이를 224로 적용해서 이미지 사이지를 수정합니다.
        """
        daily_size = len(self._df["Daily"].unique())
        gender_size = len(self._df["Gender"].unique())
        emb_size = len(self._df["Embellishment"].unique())ㅁ

        row = self._df.iloc[i]

        img_name = row["image_name"]
        img = io.imread(os.path.join(self._dir_path,img_name))
        if img.shape[2] != 3:
            img = color.rgba2rgb(img)

        x_min = int(row["BBox_xmin"])
        y_min = int(row["BBox_ymin"])
        x_max = int(row["BBox_xmax"])
        y_max = int(row["BBox_ymax"])

        height = y_max - y_min
        width = x_max - x_min

        img_ = copy.deepcopy(img)
        img_ = img_[y_min:y_max,x_min:x_max]

        if height != width :
            if height > width :
                new_h = self._img_size
                new_w = int(width * self._img_size / height)
            else :
                new_w = self._img_size
                new_h = int(height * self._img_size / width)
        else :
            new_h = self._img_size 
            new_w = self._img_size
        
        img_ = transform.resize(img_, (new_h, new_w), mode='constant')

        daily = np.zeros(daily_size)
        gender = np.zeros(gender_size)
        emb = np.zeros(emb_size)

        daily[row["Daily"]] = 1.0
        gender[row["Gender"]] = 1.0
        emb[row["Embellishment"]] = 1.0
        data = {"image" : img_, "daily" : daily, "gender" : gender, "embellishment" : emb}
        return data


    def get_dataset_parallel(self) :
        """
        get_data 함수를 병렬적으로 수행해서 전체 데이터를 불러옵니다.
        """
        num_cores = multiprocessing.cpu_count()
        dataset = parmap.map(self.get_data, 
            range(len(self._df)), 
            pm_pbar=True, 
            pm_processes=num_cores
        )
        return dataset


    def get_dataset(self) :
        """
        get_data 함수를 순차적으로 수행해서 전체 데이터를 불러옵니다.
        """
        dataset = []
        for i in tqdm(range(len(self._df))) :
            data = self.get_data(i)
            dataset.append(data)
        return dataset


    def get_label_size(self) :
        """
        label에 해당되는 Daily, Gender, Embellishment의 사이즈를 파악해서 반환합니다.
        """
        daily_size = len(self._df["Daily"].unique())
        gender_size = len(self._df["Gender"].unique())
        emb_size = len(self._df["Embellishment"].unique())
        
        label_size = {"daily" : daily_size, "gender" : gender_size, "embellishment" : emb_size}
        return label_size