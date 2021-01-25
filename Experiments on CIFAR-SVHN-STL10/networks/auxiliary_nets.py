
import torch.nn as nn
import torch.nn.functional as F
from .losses import SupConLoss


class Decoder(nn.Module):
    def __init__(self, inplanes, image_size, interpolate_mode='bilinear', widen=1):
        super(Decoder, self).__init__()

        self.image_size = image_size

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.bce_loss = nn.BCELoss()

        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, int(12 * widen), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(12 * widen)),
            nn.ReLU(),
            nn.Conv2d(int(12 * widen), 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features, image_ori):
        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='nearest')
        else:
            raise NotImplementedError

        return self.bce_loss(self.decoder(features), image_ori)


class AuxClassifier(nn.Module):
    def __init__(self, inplanes, net_config='1c2f', loss_mode='contrast', class_num=10, widen=1, feature_dim=128):
        super(AuxClassifier, self).__init__()

        assert inplanes in [16, 32, 64]
        assert net_config in ['0c1f', '0c2f', '1c1f', '1c2f', '1c3f', '2c2f']
        assert loss_mode in ['contrast', 'cross_entropy']

        self.loss_mode = loss_mode
        self.feature_dim = feature_dim

        if loss_mode == 'contrast':
            self.criterion = SupConLoss()
            self.fc_out_channels = feature_dim
        elif loss_mode == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            self.fc_out_channels = class_num
        else:
            raise NotImplementedError

        if net_config == '0c1f':  # Greedy Supervised Learning (Greedy SL)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(inplanes, self.fc_out_channels),
            )

        if net_config == '0c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(16, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '1c1f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), self.fc_out_channels),
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), self.fc_out_channels),
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), self.fc_out_channels),
                )

        if net_config == '1c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '1c3f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '2c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(32 * widen), int(32 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

    def forward(self, x, target):

        features = self.head(x)

        if self.loss_mode == 'contrast':
            assert features.size(1) == self.feature_dim
            features = F.normalize(features, dim=1)
            features = features.unsqueeze(1)
            loss = self.criterion(features, target, temperature=0.07)
        elif self.loss_mode == 'cross_entropy':
            loss = self.criterion(features, target)
        else:
            raise NotImplementedError

        return loss
