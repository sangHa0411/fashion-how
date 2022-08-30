import numpy as np
from tqdm import tqdm

class Preprocessor :

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