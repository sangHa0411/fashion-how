import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet152


class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if label is not None:
            output = (label * phi) + ((1.0 - label) * cosine)
        else:
            output = cosine
        output *= self.s
        return output

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
        self._arc1 = ArcMarginProduct(self._feature_size, self._class1_size)
        self._arc2 = ArcMarginProduct(self._feature_size, self._class2_size)
        self._arc3 = ArcMarginProduct(self._feature_size, self._class3_size)

    def forward(self, x, y=None):
        h = self._resnet(x)
        h = self._drop(h)

        if y is not None  :
            y1, y2, y3 = y
            
            o1 = self._arc1(h, y1)
            o2 = self._arc2(h, y2)
            o3 = self._arc3(h, y3)
        else :
            o1 = self._arc1(h)
            o2 = self._arc2(h)
            o3 = self._arc3(h)

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
