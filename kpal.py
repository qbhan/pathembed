import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_models import *


# class resBlock(nn.Module):
#     def __init__(self, in_ch, hid_ch):
#         super(resBlock, self).__init__()
#         self.act = nn.ReLU()
#         self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1)

#     def forward(self, x):
#         return x + self.conv2(self.act(self.conv1(self.act(x))))


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




# model = singleFrameDenoiser(34, 100, kernel_size=21)
# model = multiScaleDenoiser(34, 50)
# model.to('cuda')
# inp = torch.zeros((4, 34, 128, 128)).to('cuda')
# print(model(inp).shape)