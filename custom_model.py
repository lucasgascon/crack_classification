# %%

import timm
from torch import nn
import torch
import torchvision

from torchsummary import summary




class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            # 'resnet50',
            'efficientnet_b0',
            pretrained=True,
            )

        self.linear = nn.Linear(1000, 1)

    def forward(self, x):
        hidden = self.backbone(x)
        pred = self.linear(hidden)
        return pred

class CrackClassifier(nn.Module):
    def __init__(self, device):
        super(CrackClassifier,self).__init__()
        self.resnet = timm.create_model(
            'resnet50',
            pretrained=True)
        self.resnet_fc = nn.Sequential(
               nn.Linear(1000, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 1),
               nn.LogSoftmax(dim=1)).to(device)

    def forward(self,x):
        x = self.resnet(x)
        x = self.resnet_fc(x)
        return x

class Net16(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=False).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)
        
        # self.batchnorm = nn.BatchNorm2d(512)
        # self.dropout = nn.Dropout(p=.1)
        self.flat = nn.Flatten()

        self.fc = nn.Linear(138240,1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        # x = self.batchnorm(conv5)
        # x = self.dropout(x)
        x = self.flat(conv5)
        x = self.fc(x)

        return x


def load_net_vgg16():
    model = Net16()

    pretrained_dict = torch.load('models/model_unet_vgg_16_best.pt', map_location=torch.device('cpu'))['model']

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 

    # 3. load the new state dict
    # model.load_state_dict(pretrained_dict)
    model.load_state_dict(model_dict)


    # Freeze first layers
    model.conv1.requires_grad_(False)
    model.conv2.requires_grad_(False)
    model.conv3.requires_grad_(False)
    model.conv4.requires_grad_(False)
    model.conv5.requires_grad_(False)


    return model


model = load_net_vgg16()
summary(model, (3,250,300))


def unfreeze(model, epoch) : 

    if epoch == 0 : 
        model.conv1.requires_grad_(False)
        model.conv2.requires_grad_(False)
        model.conv3.requires_grad_(False)
        model.conv4.requires_grad_(False)
        model.conv5.requires_grad_(False)

    if epoch == 25 : 
        model.conv1.requires_grad_(False)
        model.conv2.requires_grad_(False)
        model.conv3.requires_grad_(False)
        model.conv4.requires_grad_(False)
        model.conv5.requires_grad_(True)

    if epoch == 30 : 
        model.conv1.requires_grad_(False)
        model.conv2.requires_grad_(False)
        model.conv3.requires_grad_(False)
        model.conv4.requires_grad_(True)
        model.conv5.requires_grad_(True)

    
    if epoch == 35 : 
        model.conv1.requires_grad_(False)
        model.conv2.requires_grad_(False)
        model.conv3.requires_grad_(True)
        model.conv4.requires_grad_(True)
        model.conv5.requires_grad_(True)

    if epoch == 40 : 
        model.conv1.requires_grad_(True)
        model.conv2.requires_grad_(True)
        model.conv3.requires_grad_(True)
        model.conv4.requires_grad_(True)
        model.conv5.requires_grad_(True)

    return model
# %%