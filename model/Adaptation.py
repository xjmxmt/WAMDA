import torch
import torch.nn as nn
from util.OfficeHomeDataset import OfficeHomeDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


class Adaption(nn.Module):
    def __init__(self, f_dim=256, c_dim=256, n_classes=3):
        super(Adaption, self).__init__()
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.n_classes = n_classes

        # Get ResNet50 model
        ResNet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        ResNet50 = list(ResNet50.children())[:-1]  # Remove the last FC(2048, 1000)
        self.ResNet50 = nn.Sequential(*ResNet50)

        self.sourceEncoder = nn.Sequential(
            nn.Linear(f_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # expect 2-D input
            nn.ELU(),
            nn.Linear(1024, self.c_dim),
            nn.BatchNorm1d(self.c_dim),
            nn.ELU(),
            nn.Linear(self.c_dim, self.c_dim),
            nn.BatchNorm1d(self.c_dim),
            nn.ELU()
        )

        self.targetEncoder = nn.Sequential(
            ResNet50(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.linear(1024, c_dim),
            nn.BatchNorm1d(c_dim),
            nn.ELU(),
            nn.linear(c_dim, c_dim),
            nn.BatchNorm1d(c_dim),
            nn.ELU()
        )
        self.targetClassifier = nn.Sequential(
            nn.Linear(self.c_dim, self.c_dim),
            nn.ELU(),
            nn.Linear(self.c_dim, n_classes)
        )


    def forward(self, input_batch):
        source_encoder = self.sourceEncoder(input_batch)
        target_encoder = self.targetEncoder(input_batch)
        target_classifier = self.targetClassifier(target_encoder)
        return source_encoder, target_encoder, target_classifier