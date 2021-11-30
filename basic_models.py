import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class ResBlock(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super(ResBlock, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.act(self.conv2(self.act(self.conv1(x))))


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print('up', x1.shape)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=64, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, hidden)
        self.down1 = Down(hidden, hidden)
        self.down2 = Down(hidden, hidden*2)
        self.down3 = Down(hidden*2, hidden*4)
        self.down4 = Down(hidden*4, hidden*8 // factor)

        self.up1 = Up(hidden*8, hidden*4 // factor, bilinear)
        self.up2 = Up(hidden*4, hidden*2 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(hidden*2, hidden, bilinear)
        self.up4 = Up(hidden*2, hidden, bilinear)
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print('inc x1', x1.shape)
        x2 = self.down1(x1)
        # print('down x2', x2.shape)
        x3 = self.down2(x2)
        # print('down x3', x3.shape)
        x4 = self.down3(x3)
        # print('down x4', x4.shape)
        x5 = self.down4(x4)
        # print('down x5', x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # print('up x3', x.shape)
        x = self.up3(x, x2)
        # print('up x2', x.shape)
        x = self.up4(x, x1)
        # print('up x1', x.shape)
        x = self.outc(x)
        return x


class UNet_Half(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=64, bilinear=True):
        super(UNet_Half, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 1 if bilinear else 1

        self.inc = DoubleConv(n_channels, hidden)
        self.down1 = Down(hidden, hidden)
        self.down2 = Down(hidden, hidden*2)
        self.down3 = Down(hidden*2, hidden*4)
        # self.down4 = Down(hidden*4, hidden*8 // factor)

        # self.up1 = Up(hidden*8, hidden*4 // factor, bilinear)
        self.up2 = Up(hidden*6, hidden*2 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(hidden*3, hidden, bilinear)
        self.up4 = Up(hidden*2, hidden, bilinear)
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print('inc x1', x1.shape)
        x2 = self.down1(x1)
        # print('down x2', x2.shape)
        x3 = self.down2(x2)
        # print('down x3', x3.shape)
        x4 = self.down3(x3)
        # print('down x4', x4.shape)
        # x5 = self.down4(x4)
        # print('down x5', x5.shape)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        # print('up x3', x.shape)
        x = self.up3(x, x2)
        # print('up x2', x.shape)
        x = self.up4(x, x1)
        # print('up x1', x.shape)
        # logits = self.outc(x)
        return x


# class UNet_quad(nn.Module):
#     def __init__(self, n_channels, n_classes, hidden=64, bilinear=True):
#         super(UNet_Half, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         factor = 2 if bilinear else 1

#         self.inc = DoubleConv(n_channels, hidden)
#         self.down1 = Down(hidden, hidden)
#         self.down2 = Down(hidden, hidden*2)
#         self.down3 = Down(hidden*2, hidden*4)
#         self.down4 = Down(hidden*4, hidden*8 // factor)

#         self.up1 = Up(hidden*8, hidden*4 // factor, bilinear)
#         self.up2 = Up(hidden*4, hidden*2 // factor, bilinear)
#         # self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(hidden*2, hidden, bilinear)
#         self.up4 = Up(hidden*2, hidden, bilinear)
#         self.outc = OutConv(hidden, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         # print('inc x1', x1.shape)
#         x2 = self.down1(x1)
#         # print('down x2', x2.shape)
#         x3 = self.down2(x2)
#         # print('down x3', x3.shape)
#         x4 = self.down3(x3)
#         # print('down x4', x4.shape)
#         x5 = self.down4(x4)
#         # print('down x5', x5.shape)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         # print('up x3', x.shape)
#         x = self.up3(x, x2)
#         # print('up x2', x.shape)
#         x = self.up4(x, x1)
#         # print('up x1', x.shape)
#         logits = self.outc(x)
#         return logits

# net = UNet(34, 3, hidden=64)
# print('# Parameter for DecompNet : {}'.format(sum([p.numel() for p in net.parameters()])))
# net_half = UNet_Half(34, 3, bilinear=True)
# print('# Parameter for DecompNet : {}'.format(sum([p.numel() for p in net_half.parameters()])))
# batch = torch.rand((8, 34, 128, 128))
# ret = net(batch)
# print(ret.shape)