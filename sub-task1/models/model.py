import torch
import torch.nn as nn
from torchvision.models.resnet import resnet152
from torchvision.models.densenet import densenet161

class ResidualClassifier(nn.Module) :

    def __init__(self, feature_size, hidden_size, class_size) :
        super(ResidualClassifier, self).__init__()

        self._feature_size = feature_size
        self._hidden_size = hidden_size
        self._class_size = class_size

        self._long = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size)
        )

        self._short = nn.Linear(self._feature_size, self._hidden_size)
        self._act = nn.ReLU()
        self._classifier = nn.Linear(self._hidden_size, self._class_size)

    def forward(self, x) :
        h1 = self._long(x)
        h2 = self._short(x)

        h = self._act(h1 + h2)
        o = self._classifier(h)
        return o



class ResNetBasedModel(nn.Module):
    def __init__(self, 
        hidden_size,
        class1_size, 
        class2_size, 
        class3_size,
        dropout_prob,
        pretrained=True
    ):
        super(ResNetFeedForwardModel, self).__init__()

        if pretrained == True :
            backbone = resnet152(pretrained=True, progress=True)
            self._backbone = nn.Sequential(*list(backbone.children())[:-1])
        else :
            backbone = resnet152(pretrained=False, progress=False)
            self._backbone = nn.Sequential(*list(backbone.children())[:-1])


        self._feature_size = 2048
        self._hidden_size = hidden_size
        self._dropout_prob = dropout_prob
        self._class1_size = class1_size
        self._class2_size = class2_size
        self._class3_size = class3_size

        self._drop = nn.Dropout(self._dropout_prob)
        self._classifier1 = nn.Linear(self._feature_size, self._class1_size)
        self._classifier2 = nn.Linear(self._feature_size, self._class2_size)
        self._classifier3 = nn.Linear(self._feature_size, self._class3_size)


    def forward(self, x):
        h = self._backbone(x)
        h = h.squeeze()
        h = self._drop(h)

        y1 = self._classifier1(h)
        y2 = self._classifier2(h)
        y3 = self._classifier3(h)
        return y1, y2, y3



class ResNetFeedForwardModel(nn.Module):
    def __init__(self, 
        hidden_size,
        class1_size, 
        class2_size, 
        class3_size,
        dropout_prob,
        pretrained=True
    ):
        super(ResNetFeedForwardModel, self).__init__()

        if pretrained == True :
            backbone = resnet152(pretrained=True, progress=True)
            self._backbone = nn.Sequential(*list(backbone.children())[:-1])
        else :
            backbone = resnet152(pretrained=False, progress=False)
            self._backbone = nn.Sequential(*list(backbone.children())[:-1])


        self._feature_size = 2048
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
        h = h.squeeze()
        h = self._drop(h)

        y1 = self._classifier1(h)
        y2 = self._classifier2(h)
        y3 = self._classifier3(h)
        return y1, y2, y3
