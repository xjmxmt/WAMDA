import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from random import sample


class OfficeHomeDataset(Dataset):
    def __init__(self, data_path, domain=None, balance=False, transform=None):
        self.transform = transform
        self.domain = domain
        self.balance = balance

        # label dict
        self.label_dict = {"Art": 0, "Clipart":1, "Product":2}

        # Read all file names
        self.file_names = []
        if self.domain is None:
            for root, dirs, files in os.walk(data_path):
                for filename in files:
                    if filename == ".DS_Store": continue
                    self.file_names.append(os.path.join(root, filename))
        else:
            domain_file = []
            source_file = []
            for root, dirs, files in os.walk(data_path):
                if self.domain in root:
                    for filename in files:
                        if filename == ".DS_Store": continue
                        domain_file.append(os.path.join(root, filename))
                else:
                    for filename in files:
                        if filename == ".DS_Store": continue
                        source_file.append(os.path.join(root, filename))
            if balance:
                self.file_names = domain_file + sample(source_file, len(domain_file))
            else:
                self.file_names = domain_file + source_file

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = []
        filename = self.file_names[idx]
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        # print(img.shape, filename)
        source_name = filename.split('/')[-3]
        if self.domain is None:
            label.append(self.label_dict[source_name])
        else:
            print("name: ", source_name)
            if source_name == self.domain:
                label.append(1)
            else: label.append(0)
        label = np.array(label)
        sample = {'image': img, 'label': label}

        return sample


if __name__ == "__main__":
    # # Read image for ResNet50
    dataset = OfficeHomeDataset("../dataset", domain="Real World", balance=True, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print(len(dataset.file_names))
    print(dataset.file_names)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'].size(), sample_batched['label'])

    # for inputs, labels in dataloader['train']:
    #     print(inputs.shape, labels.shape)