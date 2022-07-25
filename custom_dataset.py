# %%

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


image_T = {
    "train": T.Compose([
        T.RandomResizedCrop(
            size = (250,300),
            scale = (0.8,1),
            ratio = (0.75, 1.33),
        ),
        T.ToTensor(),
        T.RandomHorizontalFlip(p=.5),
        T.RandomVerticalFlip(p=.5),
        T.RandomRotation(degrees = (0,180)),
        T.ColorJitter(
            brightness=.1,
            saturation  = (.3,.7),
            hue = .05,
        ),
        T.RandomPerspective(
            distortion_scale=0.6,
            p=1.0,
        ),
        T.RandomAutocontrast(p=.2),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
    "valid": T.Compose([
        T.Resize((250, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) 
}


#%%



"""from tqdm import tqdm 
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

TRAIN_DATA_FOLDER = "data/images-sep/train"
BATCH_SIZE = 32
NB_EPOCHS = 20
NUM_WORKER = 0



train_dataset = ImageFolder(
    root=TRAIN_DATA_FOLDER, 
    transform=image_T['train']
    )

# compute train samples_weights
train_counts = np.bincount(train_dataset.targets)
train_class_weights = 1. / train_counts
train_samples_weights = train_class_weights[train_dataset.targets]

train_sampler = WeightedRandomSampler(
    weights=train_samples_weights,
    num_samples=len(train_samples_weights),
    replacement=False)


train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler = train_sampler, 
    num_workers=NUM_WORKER,
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler = train_sampler, 
    num_workers=NUM_WORKER,
)

from torchvision.utils import make_grid

epoch = 0
for i, (input, target) in enumerate(tqdm(train_dataloader)):

        if (i < 1) and (epoch == 0):
            grid = make_grid(input)
            plt.imshow(grid)"""