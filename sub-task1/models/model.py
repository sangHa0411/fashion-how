import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet152

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s = 10, m = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels=None):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        if labels == None :
            return cos_th

        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m

        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        outputs = labels * cos_th_m + (1.0 - labels) * cos_th
        outputs = outputs * self.s
        return outputs


class ArcFaceModel(nn.Module):
    def __init__(self, 
        hidden_size,
        class1_size, 
        class2_size, 
        class3_size,
        dropout_prob,
        pretrained=True
    ):
        super(ArcFaceModel, self).__init__()

        if pretrained == True :
            self._resnet = resnet152(pretrained=True, progress=True)
        else :
            self._resnet = resnet152(pretrained=False, progress=False)

        self._feature_size = 1000
        self._hidden_size = hidden_size
        self._dropout_prob = dropout_prob
        self._class1_size = class1_size
        self._class2_size = class2_size
        self._class3_size = class3_size

        self._drop = nn.Dropout(self._dropout_prob)
        self._backbone1 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.BatchNorm1d(self._hidden_size)
        )

        self._backbone2 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.BatchNorm1d(self._hidden_size)
        )

        self._backbone3 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.BatchNorm1d(self._hidden_size)
        )

        self._arc1 = ArcModule(self._hidden_size, self._class1_size)
        self._arc2 = ArcModule(self._hidden_size, self._class2_size)
        self._arc3 = ArcModule(self._hidden_size, self._class3_size)

    def forward(self, x, y=None):
        h = self._resnet(x)
        h = self._drop(h)

        h1 = self._backbone1(h)
        h2 = self._backbone2(h)
        h3 = self._backbone3(h)

        if y is not None :
            y1, y2, y3 = y
            
            o1 = self._arc1(h1, y1)
            o2 = self._arc2(h2, y2)
            o3 = self._arc3(h3, y3)
        else :
            o1 = self._arc1(h1)
            o2 = self._arc2(h2)
            o3 = self._arc3(h3)

        return o1, o2, o3


class BaseModel(nn.Module):
    def __init__(self, 
        hidden_size,
        class1_size, 
        class2_size, 
        class3_size,
        dropout_prob,
        pretrained=True
    ):
        super(BaseModel, self).__init__()

        if pretrained == True :
            self._backbone = resnet152(pretrained=True, progress=True)
        else :
            self._backbone = resnet152(pretrained=False, progress=False)

        self._feature_size = 1000
        self._hidden_size = hidden_size
        self._dropout_prob = dropout_prob
        self._class1_size = class1_size
        self._class2_size = class2_size
        self._class3_size = class3_size

        self._drop = nn.Dropout(self._dropout_prob)
        self._classifier1 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._class1_size)
        )

        self._classifier2 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._class2_size)
        )

        self._classifier3 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._class3_size)
        )

    def forward(self, x):
        h = self._backbone(x)
        h = self._drop(h)

        y1 = self._classifier1(h)
        y2 = self._classifier2(h)
        y3 = self._classifier3(h)
        return y1, y2, y3
