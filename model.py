import torch.nn as nn
import math

class Generator(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.model = nn.Sequential(
            nn.Linear(self.params["LATENT_DIM"], 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, math.prod(self.params["IMG_SHAPE"])),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self.params["IMG_SHAPE"])
        return img


class Discriminator(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.model = nn.Sequential(
            nn.Linear(math.prod(self.params["IMG_SHAPE"]), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.params["OUTPUT_DIM"]),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flatten = img.view(img.size(0), -1)
        validity = self.model(img_flatten)
        return validity
