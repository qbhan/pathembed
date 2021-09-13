import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_models import *
from utils import *


class scaleCompositor(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super(scaleCompositor, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(hid_ch, 1, kernel_size=1)
        self.resblock1 = resBlock(hid_ch, hid_ch)
        self.resblock2 = resBlock(hid_ch, hid_ch)
        self.act = nn.Sigmoid()
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, f, c):
        # print(f.shape, c.shape)
        x = torch.cat((f, c), dim=1)
        # print(x.shape)
        UDf = self.upsample(self.downsample(f))
        scale = self.conv2(self.resblock2(self.resblock1(self.conv1(x))))
        scale = self.act(scale)
        # print(scale.shape)
        return f - torch.matmul(scale, UDf) + torch.matmul(scale, c)


class multiScaleDenoiser(nn.Module):
    def __init__(self, hid_ch, models):
        super(multiScaleDenoiser, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.denoiser1 = models[0]
        self.denoiser2 = models[1]
        self.denoiser3 = models[2]
        self.compositor1 = scaleCompositor(3 * 2, hid_ch)
        self.compositor2 = scaleCompositor(3 * 2, hid_ch)

        # grad off for denoisers
        for param in self.denoiser1:
            param.requires_grad = False

        for param in self.denoiser2:
            param.requires_grad = False

        for param in self.denoiser3:
            param.requires_grad = False

    def _load_denoiser(self, pth):
        load_model = torch.load(pth)
        self.denoiser1.load_state_dict(load_model['model_state_dict'])
        self.denoiser2.load_state_dict(load_model['model_state_dict'])
        self.denoiser3.load_state_dict(load_model['model_state_dict'])
        # optimizerDiff.load_state_dict(checkpointDiff['optimizer_state_dict'])

    def forward(self, x):
        down1 = self.downsample(x)
        down2 = self.downsample(down1)
        print('input :', x.shape)
        print('input downsampled 1:', down1.shape)
        print('input downsampled 2:', down2.shape)

        denoise1 = self.denoiser1(x)
        denoise2 = self.denoiser2(down1)
        denoise3 = self.denoiser3(down2)

        upscale1 = self.compositor1(denoise2, self.upsample(denoise3))
        upscale2 = self.compositor2(denoise1, self.upsample(upscale1))

        return upscale2