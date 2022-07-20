from torchvision import transforms


TRANSFORMS = [
    transforms.ToTensor(),
    # TODO: add scaling here
    transforms.RandomCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
]


class CustomDataset():
    def __init__(self, path: str):
        raise NotImplementedError
        self.path = path
        self.img_path, self.json_path = get_img_annots_folders(path)
        img_names = os.listdir(self.img_path)
        img_names = [name[:-4] for name in img_names]
        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        raise NotImplementedError

        # aller chercher l'image idx (PIL)
        # appliquer toutes les transforms
        for transform in TRANSFORMS:
            img = transform(img)

        img_name = self.img_names[idx]
        img_path = self.img_path + img_name + '.jpg'
        json_path = self.json_path + img_name + '.jpg.json'

        with open(json_path, 'r') as json_file:
            labels_dict = json.load(json_file)

        if labels_dict is None:
            print("labels_dict is None")
        if len(labels_dict['tags']) != 0:
            print("Replacing tags")
            labels_dict['tags'] = []
        return img, label