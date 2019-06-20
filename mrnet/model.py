#!/usr/bin/python3

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = self._create_feature_extractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.classifier = nn.Linear(256, 1)

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()
            out = torch.sigmoid(self.classifier(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out

    @staticmethod
    def _create_feature_extractor():
        model = models.alexnet(pretrained=True).features

        for param in model.parameters():
            param.requires_grad = False

        return model
