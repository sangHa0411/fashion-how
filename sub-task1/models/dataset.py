import copy
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, i):
        data = self.dataset[i]

        img = data["image"]
        daily = data["daily"]
        gender = data["gender"]
        emb = data["embellishment"]
    
        img_ = copy.deepcopy(img)

        img_ = self.to_tensor(img_)
        img_ = self.normalize(img_)
        img_ = img_.float()

        ret = {}
        ret['image'] = img_
        ret['daily'] = daily
        ret['gender'] = gender
        ret['embellishment'] = emb
        return ret

    def __len__(self):
        return len(self.dataset)