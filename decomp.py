import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_models import *
from utils import *
from kpcn import *

class decompModule(nn.Module):
    def __init__(self, in_channel=34, out_channel=3):
        super(decompModule, self).__init__()
        self.unet = UNet(in_channel, out_channel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.unet(x)
        mask = self.act(out)
        # print(mask + (torch.ones_like(mask) - mask))

        d1 = x[:, :3, :, :] @ mask
        d2 = x[:, :3, :, :] @ (torch.ones_like(mask) - mask)

        f1 = torch.cat((d1, x[:, 3:, :, :]), dim=1)
        f2 = torch.cat((d2, x[:, 3:, :, :]), dim=1)

        return mask, f1, f2


# decomp = decompModule()
# data = torch.randn((8, 34, 128, 128))
# mask, ret1, ret2 = decomp(data)
# print(ret1.shape)


class decompDenoiser(nn.Module):
    def __init__(self, n_layers, in_channel, hidden_channel, kernel_size, mode, pth=None):
        super(decompDenoiser, self).__init__()
        self.decomp1 = decompModule()
        self.decomp2 = decompModule()
        self.decomp3 = decompModule()
        self.denoiser1 = None
        # self.denoiser2 = None
        # self.denoiser3 = None
        # self.denoiser4 = None
        if 'kpcn' in mode and pth is not None:
            self.denoiser1 = KPCN(n_layers, in_channel, hidden_channel, kernel_size)
            # self.denoiser2 = make_net(n_layers, in_channel, hidden_channel, kernel_size, mode)
            # self.denoiser3 = make_net(n_layers, in_channel, hidden_channel, kernel_size, mode)
            # self.denoiser4 = make_net(n_layers, in_channel, hidden_channel, kernel_size, mode)
            denoiser_state = torch.load(pth)
            self.denoiser1.load_state_dict(denoiser_state['model_state_dict'])
            # self.denoiser2.load_state_dict(denoiser_state['model_state_dict'])
            # self.denoiser3.load_state_dict(denoiser_state['model_state_dict'])
            # self.denoiser4.load_state_dict(denoiser_state['model_state_dict'])
        
            for param in self.denoiser1.parameters():
                param.requires_grad = False

    def forward(self, x):
        mask1, f1, f2 = self.decomp1(x)
        mask2, f11, f12 = self.decomp2(f1)
        mask3, f21, f22 = self.decomp3(f2)

        k11 = self.denoiser1(f11)
        # k12 = self.denoiser2(f12)
        # k21 = self.denoiser3(f21)
        # k22 = self.denoiser4(f22)
        k12 = self.denoiser1(f12)
        k21 = self.denoiser1(f21)
        k22 = self.denoiser1(f22)
        # print('kernel size: ', k22.shape)

        i11 = crop_like(f11, k11)
        d11 = apply_kernel(k11, i11, 'cuda')
        i12 = crop_like(f12, k12)
        d12 = apply_kernel(k12, i12, 'cuda')
        i21 = crop_like(f21, k21)
        d21 = apply_kernel(k21, i21, 'cuda')
        i22 = crop_like(f22, k22)
        d22 = apply_kernel(k22, i22, 'cuda')

        mask1 = crop_like(mask1, d11)
        mask2 = crop_like(mask2, d11)
        mask3 = crop_like(mask3, d11)
        
        d1 = torch.div(d11, mask2) + torch.div(d12, (torch.ones_like(mask2) - mask2))
        d2 = torch.div(d21, mask3) + torch.div(d22, (torch.ones_like(mask3) - mask3))
        return torch.div(d1, mask1) + torch.div(d1, (torch.ones_like(mask1) - mask1))
        



# diffuseNet = decompDenoiser(9, 34, 100, 5, 'kpcn').to('cuda')
# x = torch.ones((8, 34, 128, 128)).to('cuda')
# out1 = diffuseNet(x)
# print(out1)