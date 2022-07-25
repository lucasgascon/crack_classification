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