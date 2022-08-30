
import os
import copy
import numpy as np
import pandas as pd
import multiprocessing
import parmap
from skimage import io, transform, color
from tqdm import tqdm

class Loader :

    def __init__(self, info_path, dir_path, img_size) :
        self._info_path = info_path
        self._dir_path = dir_path
        self._img_size = img_size
        self._df = pd.read_csv(info_path)

    def get_data(self, i) :

        daily_size = len(self._df["Daily"].unique())
        gender_size = len(self._df["Gender"].unique())
        emb_size = len(self._df["Embellishment"].unique())

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
        daily[row["Daily"]] = 1.0

        gender = np.zeros(gender_size)
        gender[row["Gender"]] = 1.0

        emb = np.zeros(emb_size)
        emb[row["Embellishment"]] = 1.0

        data = {"image" : img_, "daily" : daily, "gender" : gender, "embellishment" : emb}
        return data

    def get_dataset(self) :
        num_cores = multiprocessing.cpu_count()
        dataset = parmap.map(self.get_data, 
            range(len(self._df)), 
            pm_pbar=True, 
            pm_processes=num_cores
        )
        return dataset

    def get_label_size(self) :

        daily_size = len(self._df["Daily"].unique())
        gender_size = len(self._df["Gender"].unique())
        emb_size = len(self._df["Embellishment"].unique())
        
        label_size = {"daily" : daily_size, "gender" : gender_size, "embellishment" : emb_size}
        return label_size