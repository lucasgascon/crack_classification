# %%

import timm
from torch import nn
import torch
import torchvision

from torchsummary import summary


class Net16(nn.Module):
    def __init__(self):
        super().__init__()


        # self.num_classes = 1

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
        

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        pred = conv5

        return pred

#%%

def load_net_vgg16():
    model = Net16()

    # checkpoint = torch.load('models/model_unet_vgg_16_best.pt', map_location=torch.device('cpu'))
    # sub_keys = ('encoder.0.weight', 'encoder.0.bias', 'encoder.2.weight', 'encoder.2.bias', 'encoder.5.weight', 'encoder.5.bias', 'encoder.7.weight', 'encoder.7.bias', 'encoder.10.weight', 'encoder.10.bias', 'encoder.12.weight', 'encoder.12.bias', 'encoder.14.weight', 'encoder.14.bias', 'encoder.17.weight', 'encoder.17.bias', 'encoder.19.weight', 'encoder.19.bias', 'encoder.21.weight', 'encoder.21.bias', 'encoder.24.weight', 'encoder.24.bias', 'encoder.26.weight', 'encoder.26.bias', 'encoder.28.weight', 'encoder.28.bias', 'conv1.0.weight', 'conv1.0.bias', 'conv1.2.weight', 'conv1.2.bias', 'conv2.0.weight', 'conv2.0.bias', 'conv2.2.weight', 'conv2.2.bias', 'conv3.0.weight', 'conv3.0.bias', 'conv3.2.weight', 'conv3.2.bias', 'conv3.4.weight', 'conv3.4.bias', 'conv4.0.weight', 'conv4.0.bias', 'conv4.2.weight', 'conv4.2.bias', 'conv4.4.weight', 'conv4.4.bias', 'conv5.0.weight', 'conv5.0.bias', 'conv5.2.weight', 'conv5.2.bias', 'conv5.4.weight', 'conv5.4.bias')
    # sub_checkpoint = {k:v for k, v in checkpoint['model'].items() if k in sub_keys}
    # model.load_state_dict(sub_checkpoint)

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


    model.conv5.requires_grad_(True)

    return model


# %%

model = load_net_vgg16()
summary(model, (3, 200, 350))

# %%


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=True)
        self.linear = nn.Linear(1000, 1)
        #self.softmax = nn.Softmax(dim = 1)


    def forward(self, x):
        hidden = self.backbone(x)
        pred = self.linear(hidden)
        #pred = self.softmax(x)
        return pred



# %%
