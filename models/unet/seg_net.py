import os

import torch
from torch import nn
from torchvision import models


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

vgg19_bn_path = r'models/unet/backbones/vgg19_bn-c79401a0.pth'


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers, return_feats=False):
        super(_DecoderBlock, self).__init__()
        self.return_feats = return_feats
        middle_channels = in_channels // 2
        layers1 = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers1 += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers2 = [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode1 = nn.Sequential(*layers1)
        self.decode2 = nn.Sequential(*layers2)

    def forward(self, x):
        t = self.decode1(x)
        if self.return_feats:
            return t, self.decode2(t)
        else:
            return self.decode2(t)


class SegNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=1, pretrained=True, return_feats=True):
        super(SegNet, self).__init__()
        self.return_feats = return_feats
        vgg = models.vgg19_bn()
        if n_channels != 3:
            vgg.features[0] = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if n_channels == 3 and pretrained:
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2, return_feats)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))

        if self.return_feats:
            feats, dec1 = self.dec1(torch.cat([enc1, dec2], 1))
            return feats, dec1
        else:
            dec1 = self.dec1(torch.cat([enc1, dec2], 1))
            return dec1
