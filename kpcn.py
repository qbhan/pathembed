from utils import apply_kernel, crop_like
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# recon_kernel_size = 21
recon_kernel_size = 21

import itertools
import os

class KPCN(nn.Module):
    def __init__(self, 
                n_layers, 
                input_channels, 
                hidden_channels, 
                kernel_size, 
                recon_kernel_size=21):
        super(KPCN, self).__init__()
        
        layers = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size),
            nn.ReLU(inplace=True)
        ]

        for l in range(n_layers-2):
            layers += [
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
            nn.ReLU(inplace=True)
            ] 

        layers += [nn.Conv2d(hidden_channels, recon_kernel_size**2, kernel_size)]

        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, inp=None):
        kernel = self.layers(x)
        if inp is None:
            x = crop_like(x, kernel)
            return apply_kernel(kernel, x)
        else:
            inp = crop_like(inp, kernel)
            return apply_kernel(kernel, inp)

# def make_net(n_layers, input_channels, hidden_channels, kernel_size, mode):
#   # create first layer manually
#   layers = [
#       nn.Conv2d(input_channels, hidden_channels, kernel_size),
#       nn.ReLU()
#   ]
  
#   for l in range(n_layers-2):
#     layers += [
#         nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
#         nn.ReLU()
#     ]
    
#     # params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
#     # print("Params : {}".format(params))
    
#   out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
#   layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
#   for layer in layers:
#     if isinstance(layer, nn.Conv2d):
#       nn.init.xavier_uniform_(layer.weight)
  
#   return nn.Sequential(*layers)


# def apply_kernel(weights, data, device):
#     # print('WEIGHTS: {}, DATA : {}'.format(weights.shape, data.shape))
#     # apply softmax to kernel weights
#     # print(weights.shape)
#     weights = weights.permute((0, 2, 3, 1)).to(device)
#     # print(weights.shape, data.shape)
#     _, _, h, w = data.size()
#     weights = F.softmax(weights, dim=3).view(-1, w * h, recon_kernel_size, recon_kernel_size)
#     # print(weights.shape, data.shape)
#     # now we have to apply kernels to every pixel
#     # first pad the input
#     r = recon_kernel_size // 2
#     data = F.pad(data[:,:3,:,:], (r,) * 4, "reflect")
#     # print(data.shape)
#     #print(data[0,:,:,:])
    
#     # make slices
#     R = []
#     G = []
#     B = []
#     kernels = []
#     for i in range(h):
#       for j in range(w):
#         pos = i*h+j
#         # ws = weights[:,pos:pos+1,:,:]
#         # kernels += [ws, ws, ws]
#         sy, ey = i+r-r, i+r+r+1
#         sx, ex = j+r-r, j+r+r+1
#         R.append(data[:,0:1,sy:ey,sx:ex])
#         G.append(data[:,1:2,sy:ey,sx:ex])
#         B.append(data[:,2:3,sy:ey,sx:ex])
#         #slices.append(data[:,:,sy:ey,sx:ex])
        
#     reds = (torch.cat(R, dim=1).to(device)*weights).sum(2).sum(2)
#     greens = (torch.cat(G, dim=1).to(device)*weights).sum(2).sum(2)
#     blues = (torch.cat(B, dim=1).to(device)*weights).sum(2).sum(2)
    
#     res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(device)
#     # print(res.shape)
    
#     return res
