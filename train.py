# %%

import datetime
from locale import normalize
import random
import time

import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch import nn

from custom_model import CustomModel, load_net_vgg16, unfreeze, CrackClassifier
from logs import plot_classification_report, plot_classes_preds

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.utils import make_grid

import torchvision.transforms as T

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau

import seaborn as sns

import numpy as np

import os

random.seed(24785)
torch.manual_seed(24785)

BATCH_SIZE = 32
NB_EPOCHS = 50
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
#%%

now = datetime.datetime.now()
writer_dir = "./logs/" + now.strftime('%m.%d/%H.%M') + '/'

tensorboard_writer = SummaryWriter(writer_dir)


TRAIN_DATA_FOLDER = "data/images-sep/train"
VALID_DATA_FOLDER = "data/images-sep/val"

now = datetime.datetime.now()
tmp_name = 'saved_models/leo_explo_' + now.strftime('%m_%d_%H_%M') +'.pt'


from custom_dataset import image_T

train_dataset = ImageFolder(
    root=TRAIN_DATA_FOLDER, 
    transform=image_T['train']
    )
valid_dataset = ImageFolder(
    root=VALID_DATA_FOLDER, 
    transform=image_T['valid']
    )


# compute train samples_weights
train_counts = np.bincount(train_dataset.targets)
train_class_weights = 1. / train_counts
train_samples_weights = train_class_weights[train_dataset.targets]

train_sampler = WeightedRandomSampler(
    weights=train_samples_weights,
    num_samples=len(train_samples_weights),
    replacement=False)

# compute valid samples_weights
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

# model = CustomModel().to(device)
model = load_net_vgg16().to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.005,
    betas = (0.9,0.999),
    eps = 1e-08,
    # weight_decay = 1e-3,
    amsgrad = False,
)

scheduler = ReduceLROnPlateau(optimizer, 'min')


pos_weight = torch.Tensor([train_class_weights[0] / train_class_weights[1]]).to(device)
criterion = BCEWithLogitsLoss( 
    reduction='none',
    pos_weight=pos_weight,
)

# constant for classes
classes = train_dataset.classes

best_accuracy = 0

# %%

for epoch in range(NB_EPOCHS):
    print(f'Epoch {epoch}:')
    epoch_train_losses = []
    epoch_valid_losses = []

    # model = unfreeze(model, epoch)

    model.train()

    y_train_pred = []
    y_train_true = []

    stop = time.time()
    for i, (input, target) in enumerate(tqdm(train_dataloader)):

        if (i < 1) and (epoch == 0):
            grid = make_grid(input, normalize = True)
            tensorboard_writer.add_image('images', grid, 0)
            tensorboard_writer.add_graph(model.cpu(), input)
            model = model.to(device)

        input = input.to(device)
        target = target.to(device)

        if input is None:
            continue
        start = time.time()

        output = model(input)

        output_ = (output.detach().cpu().numpy() > 0)

        y_train_pred.extend(output_)  # save prediction

        loss_per_sample = criterion(output.view(-1), target.float())
        loss = loss_per_sample.mean()

        target = target.data.cpu().numpy()
        y_train_true.extend(target)  # save ground truth

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # scheduler.step(loss)
        

        epoch_train_losses.append(loss.detach().to('cpu'))

        stop = time.time()

    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss.detach().cpu().numpy(),
    #         }, tmp_name)


    model.eval()

    y_valid_pred = []
    y_valid_true = []


    for i, (input, target) in enumerate(tqdm(valid_dataloader)):

        input = input.to(device)
        target = target.to(device)

        if input is None:
            continue
        start = time.time()

        output = model(input)

        output_ = (output.detach().cpu().numpy() > 0)
        y_valid_pred.extend(output_)  # save prediction

        loss_per_sample = criterion(output.view(-1), target.float())
        loss = loss_per_sample.mean()

        target = target.data.cpu().numpy()
        y_valid_true.extend(target)  # save ground truth

        # # Plot images with labels and predictions
        # if (i in [1,3,4,5]) and (epoch in [40,50,60,70,80,90]):
        #     figure = plot_classes_preds(input, target, output_,
        #                                 shown_batch_size=30)
        #     tensorboard_writer.add_figure('Classes preds' + str(i), figure, epoch)

        epoch_valid_losses.append(loss.detach().cpu())

        stop = time.time()

    # accuracy = accuracy_score(y_valid_true, y_valid_pred)
    # if accuracy > best_accuracy :
    #     print('Saved epoch ' + str(epoch) + ' whose accuracy is ' + str(accuracy))
    #     best_accuracy = accuracy 
    #     torch.save(model, tmp_name)


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

    tensorboard_writer.add_scalar(
        'Training epoch acc_score',
        accuracy_score(y_train_true,y_train_pred),
        epoch)

    tensorboard_writer.add_scalar(
        'Valid epoch acc_score',
        accuracy_score(y_valid_true,y_valid_pred),
        epoch)


    # Build train confusion matrix
    cf_matrix = confusion_matrix(y_train_true, y_train_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    train_heatmap = sns.heatmap(df_cm, annot=True).get_figure()
    # Save train confusion matrix to Tensorboard
    tensorboard_writer.add_figure("Train confusion matrix", train_heatmap, epoch)

    # Build valid confusion matrix
    cf_matrix = confusion_matrix(y_valid_true, y_valid_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    valid_heatmap = sns.heatmap(df_cm, annot=True).get_figure()
    # Save valid confusion matrix to Tensorboard
    tensorboard_writer.add_figure("Valid confusion matrix", valid_heatmap, epoch)

    # Precision recall curve
    train_classif_report_heatmap = plot_classification_report(y_train_true, y_train_pred)
    tensorboard_writer.add_figure("Train classif report", train_classif_report_heatmap.get_figure(), epoch)

    valid_classif_report_heatmap = plot_classification_report(y_valid_true, y_valid_pred)
    tensorboard_writer.add_figure("Valid classif report", valid_classif_report_heatmap.get_figure(), epoch)

    print(f'train_loss: {train_loss}')
    print(f'valid_loss: {valid_loss}')



tensorboard_writer.close()
# %%
