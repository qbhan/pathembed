import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import itertools
import os

from utils import *


class resBlock(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super(resBlock, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(self.act(x))))


class srcEncoder(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super(srcEncoder, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class kernelPredictor(nn.Module):
    def __init__(self, in_ch, hid_ch, pred_kernel_size=21):
        super(kernelPredictor, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(hid_ch, pred_kernel_size**2, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

    
class spatialFeatExtractor(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch=None, num_layer=24):
        super(spatialFeatExtractor, self).__init__()
        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(resBlock(in_ch, hid_ch))
            # if i == num_layer - 1:
            #     layers.append(resBlock(hid_ch, out_ch))
            else:
                layers.append(resBlock(hid_ch, hid_ch))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class singleFrameDenoiser(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=21):
        super(singleFrameDenoiser, self).__init__()
        self.encoder = srcEncoder(in_ch, hid_ch)
        self.extractor = spatialFeatExtractor(hid_ch, hid_ch)
        self.predictor = kernelPredictor(hid_ch, hid_ch, pred_kernel_size=kernel_size)
        self.kernel_size = kernel_size
    
    def forward(self, x):
        kernel = self.predictor(self.extractor(self.encoder(x)))
        kernel = F.softmax(kernel, dim=2)
        # print(kernel.shape)
        # print('kernel mean', torch.mean(kernel))
        return kernel
        # return apply_kernel_kpal(kernel, x, recon_kernel_size=self.kernel_size)


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
    def __init__(self, in_ch, hid_ch):
        super(multiScaleDenoiser, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.denoiser1 = singleFrameDenoiser(in_ch, hid_ch)
        self.denoiser2 = singleFrameDenoiser(in_ch, hid_ch)
        self.denoiser3 = singleFrameDenoiser(in_ch, hid_ch)
        self.compositor1 = scaleCompositor(3 * 2, hid_ch)
        self.compositor2 = scaleCompositor(3 * 2, hid_ch)

    def forward(self, x):
        down1 = self.downsample(x)
        down2 = self.downsample(down1)

        denoise1 = self.denoiser1(x)
        denoise2 = self.denoiser2(down1)
        denoise3 = self.denoiser3(down2)

        upscale1 = self.compositor1(denoise2, self.upsample(denoise3))
        upscale2 = self.compositor2(denoise1, self.upsample(upscale1))

        return upscale2

# model = singleFrameDenoiser(34, 100, kernel_size=21)
# model = multiScaleDenoiser(34, 50)
# model.to('cuda')
# inp = torch.zeros((4, 34, 128, 128)).to('cuda')
# print(model(inp).shape)