# %%

import datetime
import random
import time

import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from custom_model import CustomModel

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from torchvision import transforms

import numpy as np

import os


random.seed(24785)
torch.manual_seed(24785)

BATCH_SIZE = 32
NB_EPOCHS = 20
NUM_WORKER = 0

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
if(torch.backends.mps.is_available() & torch.backends.mps.is_built()): 
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device : ', device)

#device = torch.device("cpu")
#%%

writer_dir = "./logs/"

tensorboard_writer = SummaryWriter(writer_dir)

# %%


image_transforms = {
    "train": transforms.Compose([
        #transforms.Resize((288, 352)),
        transforms.RandomResizedCrop((200,250)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop(10),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
    "valid": transforms.Compose([
        transforms.Resize((200, 250)),
        transforms.ToTensor(),
    ])

}

TRAIN_DATA_FOLDER = "data/images-sep/train"
VALID_DATA_FOLDER = "data/images-sep/val"

date = datetime.datetime.now()
tmp_name = 'saved_models/leo_explo_' + datetime.datetime.strftime(date, '%H:%M:%S') +'.pt'


train_dataset = ImageFolder(
    root=TRAIN_DATA_FOLDER, 
    transform=image_transforms['train']
    )
valid_dataset = ImageFolder(
    root=VALID_DATA_FOLDER, 
    transform=image_transforms['valid']
    )


# compute train samples_weights
train_counts = np.bincount(train_dataset.targets)
train_class_weights = 1. / train_counts
train_samples_weights = train_class_weights[train_dataset.targets]

train_sampler = WeightedRandomSampler(
    weights=train_samples_weights,
    num_samples=len(train_samples_weights),
    replacement=False)

# compute train samples_weights
valid_counts = np.bincount(valid_dataset.targets)
valid_class_weights = 1. / valid_counts
valid_samples_weights = valid_class_weights[valid_dataset.targets]

valid_sampler = WeightedRandomSampler(
    weights=valid_samples_weights,
    num_samples=len(valid_samples_weights),
    replacement=False)



train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler = train_sampler, 
    num_workers=NUM_WORKER,
)

valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    sampler = valid_sampler, 
    num_workers=NUM_WORKER,
)


# %%

model = CustomModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

# %%

pos_weight = torch.Tensor([train_class_weights[0] / train_class_weights[1]]).to(device)
criterion = BCEWithLogitsLoss( 
    reduction='none',
    pos_weight=pos_weight,
)

# %%


for epoch in range(NB_EPOCHS):
    print(f'Epoch {epoch}:')
    epoch_train_losses = []
    epoch_valid_losses = []
    model.train()

    stop = time.time()
    for i, (input, target) in enumerate(tqdm(train_dataloader)):

        if i < 1:
            tensorboard_writer.add_image('test', input[0].numpy())
        
        input = input.to(device)
        target = target.to(device)

        if input is None:
            continue
        start = time.time()

        output = model(input).view(-1)

        loss_per_sample = criterion(output, target.float())
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.detach().to('cpu'))

        stop = time.time()
       
    
    torch.save(model.state_dict(), tmp_name)

    model.eval()

    for i, (input, target) in enumerate(tqdm(valid_dataloader)):

        input = input.to(device)
        target = target.to(device)

        if input is None:
            continue
        start = time.time()

        output = model(input).view(-1)

        loss_per_sample = criterion(output, target.float())
        loss = loss_per_sample.mean()

        epoch_valid_losses.append(loss.detach().to('cpu'))

        stop = time.time()

    train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    valid_loss = sum(epoch_valid_losses) / len(epoch_valid_losses)

    tensorboard_writer.add_scalar(
        'Training epoch loss',
        train_loss,
        epoch)
    tensorboard_writer.add_scalar(
        'Valid epoch loss',
        valid_loss,
        epoch)

    print(f'train_loss: {train_loss}')
    print(f'valid_loss: {valid_loss}')
# %%
