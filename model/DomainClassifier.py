import torch
import torch.nn as nn
from util.OfficeHomeDataset import OfficeHomeDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    def __init__(self, f_dim=256):
        super(DomainClassifier, self).__init__()
        self.f_dim = f_dim

        # Get ResNet50 model
        ResNet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        ResNet50 = list(ResNet50.children())[:-1]  # Remove the last FC(2048, 1000)
        self.ResNet50 = nn.Sequential(*ResNet50)

        self.sourceFeatureExtractor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # expect 2-D input
            nn.ELU(),
            nn.Linear(1024, self.f_dim),
            nn.ELU(),
            nn.Linear(self.f_dim, self.f_dim),
            nn.BatchNorm1d(self.f_dim),
            nn.ELU()
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.f_dim, int(self.f_dim/2)),
            nn.ELU(),
            nn.Linear(int(self.f_dim/2), 2),
            nn.Sigmoid()
        )


    def forward(self, input_batch):
        h1 = self.ResNet50(input_batch)
        h1 = torch.flatten(h1, start_dim=1)  # size: (batch_size, dim)
        source_feature = self.sourceFeatureExtractor(h1)
        domain_classification = self.domain_classifier(source_feature)
        return source_feature, domain_classification



if __name__ == "__main__":
    # Read image for ResNet50
    dataset = OfficeHomeDataset("../datasets", domain="Real World", balance=True, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print(len(dataset.file_names))
    print(dataset.file_names)

    from util.utils import *

    dataloaders = get_dataloaders(dataset, batch_size=4)
    model = DomainClassifier()

    crossEntropyLoss = F.cross_entropy

    optimizer = optim.Adam([
        {'params': model.ResNet50.parameters(), 'lr': 1e-5},
        {'params': model.sourceFeatureExtractor.parameters(), 'lr': 1e-4},
        {'params': model.domain_classifier.parameters(), 'lr': 1e-4}
    ])

    train_model(model, dataloaders, Binary_entropyloss, optimizer, num_epochs=3)

