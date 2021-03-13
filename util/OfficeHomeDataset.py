import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class OfficeHomeDataset(Dataset):
    def __init__(self, data_path, f_dim=256, c_dim=256, transform=None):
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.transform = transform

        # label dict
        self.label_dict = {"Art": 0, "Clipart":1, "Product":2}

        # Read all file names in __init__
        self.file_names = []
        for root, dirs, files in os.walk(data_path):
            for filename in files:
                if filename == ".DS_Store": continue
                self.file_names.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = []
        filename = self.file_names[idx]
        img = Image.open(filename)
        # img = resize(img, (self.f_dim, self.c_dim), anti_aliasing=True)
        if self.transform:
            img = self.transform(img)
        # print(img.shape, filename)
        source_name = filename.split('/')[-3]
        label.append(self.label_dict[source_name])
        label = np.array(label)
        sample = {'image': img, 'label': label}

        return sample


if __name__ == "__main__":
    # Read image for ResNet50
    dataset = OfficeHomeDataset("../dataset/Source", transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print(len(dataset.file_names))
    print(dataset.file_names)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['label'].size())

    for inputs, labels in dataloader['train']:
        print(inputs.shape, labels.shape)
