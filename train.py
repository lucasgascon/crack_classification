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
NB_EPOCHS = 10
NUM_WORKER = 0
#USE_CUDA = torch.cuda.is_available()

writer_dir = "./logs"

tensorboard_writer = SummaryWriter(writer_dir)

# %%


image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((288, 352)),
        transforms.ToTensor(),
        # TODO: add scaling here
        #transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ]),
    "valid": transforms.Compose([
        transforms.Resize((288, 352)),
        transforms.ToTensor(),
    ])

}

TRAIN_DATA_FOLDER = "data/images_split/train"
VALID_DATA_FOLDER = "data/images_split/val"

date = datetime.datetime.now()
tmp_name = 'leo_explo_' + datetime.datetime.strftime(date, '%H%M')


train_dataset = ImageFolder(
    root=TRAIN_DATA_FOLDER, 
    transform=image_transforms['train']
    )
valid_dataset = ImageFolder(
    root=VALID_DATA_FOLDER, 
    transform=image_transforms['valid']
    )


# compute samples_weights
counts = np.bincount(train_dataset.classes)
class_weights = 1. / counts
samples_weights = class_weights[train_dataset.targets]


sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=False)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler = sampler, 
    num_workers=NUM_WORKER
)

valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    sampler = sampler, 
    num_workers=NUM_WORKER
)


# %%

model = CustomModel()
optimizer = torch.optim.Adam(model.parameters())

# %%

criterion = BCEWithLogitsLoss(
    pos_weight=torch.Tensor(class_weights), 
    reduction='none',
    )

# %%


for epoch in range(NB_EPOCHS):
    print(f'Epoch {epoch}:')
    epoch_train_losses = []
    epoch_valid_losses = []
    model.train()

    stop = time.time()
    for i, (input, target) in enumerate(tqdm(train_dataloader)):

        """if i < 1:
            tensorboard_writer.add_figure('test', input[0])"""

        if input is None:
            continue
        start = time.time()

        output = model(input).view(-1)

        loss_per_sample = criterion(output, target.float())
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.detach().cpu())

        stop = time.time()
       
    torch.save(model.state_dict(), tmp_name)
    model.eval()
    for i, (input, target) in enumerate(tqdm(valid_dataloader)):

        if input is None:
            continue
        start = time.time()

        output = model(input).view(-1)

        loss_per_sample = criterion(output, target.float())
        loss = loss_per_sample.mean()

        epoch_valid_losses.append(loss.detach().cpu())

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
