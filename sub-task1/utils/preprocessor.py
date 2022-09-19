import numpy as np
from tqdm import tqdm

class Preprocessor :
    """
    높이, 너비 둘 중 길이가 긴 것이 224이기 때문에 
    작은 것에 해당되는 것의 나머지 부분을 앞뒤 혹은 양 옆으로 1.0 값으로 패딩을 합니다.
    이미지의 길이는 (224, 224, 3)으로 통일이 됩니다.
    """
    def __init__(self, img_size) :
        self._img_size = img_size

    def __call__(self, dataset) :

        for data in tqdm(dataset) :
            img = data["image"]
            height, width, _ = img.shape 

            new_img = np.ones((self._img_size, self._img_size, 3))
            if height != width :
                if width == self._img_size :
                    height_start = int((self._img_size - height) / 2)
                    new_img[height_start:height_start+height, :, :] = img
                else :
                    width_start = int((self._img_size - width) / 2)
                    new_img[:, width_start:width_start+width, :] = img
            else :
                new_img = img

            data["image"] = new_img
        
        return dataset