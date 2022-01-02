import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_models import *
from utils import *
from kpcn import *
from path import *
from path import *

class decompModule(nn.Module):
    def __init__(self, in_channel=34, out_channel=1, discrete=False, use_pbuffer=False):
        super(decompModule, self).__init__()
        # print(in_channel)
        self.unet = UNet(in_channel, out_channel)
        # self.unet = UNet_Half(in_channel, out_channel, bilinear=False)
        self.act = nn.Sigmoid()
        self.discrete = discrete
        self.out_channel = out_channel
        self.use_pbuffer = use_pbuffer
        if use_pbuffer:
            self.pathNet = PathNet(36)

    def forward(self, batch, use_llpm_buf=True):

        def _mask_spec(spec, mask):
            return torch.log((torch.exp(spec) - 1.0) * mask + 1.0)

        x = batch['kpcn_diffuse_in']
        y = batch['kpcn_specular_in']
        noisy = batch['kpcn_diffuse_buffer'] * (batch['kpcn_albedo'] + 0.00316) + torch.exp(batch['kpcn_specular_buffer']) - 1.0
        if not self.use_pbuffer and self.out_channel == 1:
            noisy = torch.norm(noisy, dim=1).unsqueeze(1)
            
        noisy = torch.log(noisy + torch.ones_like(noisy))
        # print(batch['kpcn_diffuse_in'].shape)
        noisy = torch.cat((noisy, batch['kpcn_diffuse_in'][:,10:,:,:]), dim=1)
        
        if self.use_pbuffer:
            p_buffer = self.pathNet(batch['paths'])
            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= p_buffer.shape[1]
            noisy = torch.cat((noisy, p_buffer.mean(1), p_var), dim=1)
        
        out = self.unet(noisy)
        mask = self.act(out)
        
        # mask value under 0.5 will become 0 and others will be 1
        if self.discrete: mask = (mask > 0.5).float() 
        # print(x.shape, mask.shape)
        diff_rad1 = x[:, :3, :, :] * mask
        diff_var1 = diff_rad1.var(1).unsqueeze(1)
        diff_dx1 = x[:, 4:7, :, :] * mask
        diff_dy1 = x[:, 7:10, :, :] * mask

        diff_rad2 = x[:, :3, :, :] * (torch.ones_like(mask) - mask)
        diff_var2 = diff_rad2.var(1).unsqueeze(1)
        diff_dx2 = x[:, 4:7, :, :] * (torch.ones_like(mask) - mask)
        diff_dy2 = x[:, 7:10, :, :] * (torch.ones_like(mask) - mask)
        
        spec_rad1 = torch.log((torch.exp(y[:, :3, :, :]) - 1.0) * mask + 1.0)
        spec_var1 = spec_rad1.var(1).unsqueeze(1)
        spec_dx1 = torch.log((torch.exp(y[:, 4:7, :, :]) - 1.0) * mask + 1.0)
        spec_dy1 = torch.log((torch.exp(y[:, 7:10, :, :]) - 1.0) * mask + 1.0)

        spec_rad2 = torch.log((torch.exp(y[:, :3, :, :]) - 1.0) * (torch.ones_like(mask) - mask) + 1.0)
        spec_var2 = spec_rad2.var(1).unsqueeze(1)
        spec_dx2 = torch.log((torch.exp(y[:, 4:7, :, :]) - 1.0) * (torch.ones_like(mask) - mask) + 1.0)
        spec_dy2 = torch.log((torch.exp(y[:, 7:10, :, :]) - 1.0) * (torch.ones_like(mask) - mask) + 1.0)
        # # print(d1.mean(), d2.mean())

        if use_llpm_buf:

            batch_1 = {
                'target_total': batch['target_total'] * mask,
                'target_diffuse': batch['target_diffuse'] * mask,
                'target_specular': _mask_spec(batch['target_specular'], mask),
                'kpcn_diffuse_in': torch.cat((diff_rad1, diff_var1, diff_dx1, diff_dy1, x[:, 10:-1, :, :]), dim=1),
                'kpcn_specular_in': torch.cat((spec_rad1, spec_var1, spec_dx1, spec_dy1, y[:, 10:-1, :, :]), dim=1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'] * mask,
                'kpcn_specular_buffer': _mask_spec(batch['kpcn_specular_buffer'], mask),
                'kpcn_albedo': batch['kpcn_albedo'],
                'paths': batch['paths']
            }

            batch_2 = {
                'target_total': batch['target_total'] * (torch.ones_like(mask) - mask),
                'target_diffuse': batch['target_diffuse'] * (torch.ones_like(mask) - mask),
                'target_specular': _mask_spec(batch['target_specular'], (torch.ones_like(mask) - mask)),
                'kpcn_diffuse_in': torch.cat((diff_rad2, diff_var2, diff_dx2, diff_dy2, x[:, 10:, :, :]), dim=1),
                'kpcn_specular_in': torch.cat((spec_rad2, spec_var2, spec_dx2, spec_dy2, y[:, 10:, :, :]), dim=1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'] * (torch.ones_like(mask) - mask),
                'kpcn_specular_buffer': _mask_spec(batch['kpcn_specular_buffer'], (torch.ones_like(mask) - mask)),
                'kpcn_albedo': batch['kpcn_albedo'],
                'paths': batch['paths']
            }
        else:
            batch_1 = {
                'target_total': batch['target_total'] * mask,
                'target_diffuse': batch['target_diffuse'] * mask,
                'target_specular': _mask_spec(batch['target_specular'], mask),
                'kpcn_diffuse_in': torch.cat((diff_rad1, diff_var1, diff_dx1, diff_dy1, x[:, 10:, :, :]), dim=1),
                'kpcn_specular_in': torch.cat((spec_rad1, spec_var1, spec_dx1, spec_dy1, y[:, 10:, :, :]), dim=1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'] * mask,
                'kpcn_specular_buffer': _mask_spec(batch['kpcn_specular_buffer'], mask),
                'kpcn_albedo': batch['kpcn_albedo'],
            }

            batch_2 = {
                'target_total': batch['target_total'] * (torch.ones_like(mask) - mask),
                'target_diffuse': batch['target_diffuse'] * (torch.ones_like(mask) - mask),
                'target_specular': _mask_spec(batch['target_specular'], (torch.ones_like(mask) - mask)),
                'kpcn_diffuse_in': torch.cat((diff_rad2, diff_var2, diff_dx2, diff_dy2, x[:, 10:, :, :]), dim=1),
                'kpcn_specular_in': torch.cat((spec_rad2, spec_var2, spec_dx2, spec_dy2, y[:, 10:, :, :]), dim=1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'] * (torch.ones_like(mask) - mask),
                'kpcn_specular_buffer': _mask_spec(batch['kpcn_specular_buffer'], (torch.ones_like(mask) - mask)),
                'kpcn_albedo': batch['kpcn_albedo']
            }

        for k, v in batch_1.items():
            batch_1[k] = batch_1[k].detach()
        for k, v in batch_2.items():
            batch_2[k] = batch_2[k].detach()
            
        if self.use_pbuffer:    
            return mask, batch_1, batch_2, p_buffer, noisy[:, :3]
        else:   
            return mask, batch_1, batch_2


class decompOriginModule(nn.Module):
    def __init__(self, in_channel=34, out_channel=3, discrete=False):
        super(decompOriginModule, self).__init__()
        # self.unet = UNet(in_channel, out_channel)
        self.enc = FeatureEncoder(in_channel=in_channel, out_channel=64)
        self.unet = UNet_Half(64, 64, hidden=64)
        # self.act = nn.Sigmoid()
        self.mask = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1)
        self.act = nn.Sigmoid()
        self.f1 = ResBlock(64, 64)
        self.f2 = ResBlock(64, 64)

    def forward(self, inp):
        enc = self.enc(inp)
        unet = self.unet(enc)
        mask = self.mask(unet)
        mask = self.act(mask)
        # pnet = torch.cat((unet, p_buffer), dim=1)
        f1 = self.f1(unet)
        # print(pnet.shape)
        f2 = self.f2(unet)
        return mask, f1, f2
    
    
class FeatureEncoder(nn.Module):
    def __init__(self, in_channel=34, out_channel=64):
        super(FeatureEncoder, self).__init__()
        # self.unet = UNet(in_channel, out_channel)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock(out_channel, out_channel)
        self.resblock2 = ResBlock(out_channel, out_channel)
    
    def forward(self, batch):
        return self.resblock2(self.resblock1(self.act(self.conv(batch))))

# decomp = decompModule(in_channel=25)
# data = {'target_total': torch.ones((8, 3, 128, 128)),
#         'target_diffuse': torch.ones((8, 3, 128, 128)),
#         'target_specular': torch.ones((8, 3, 128, 128)),
#         'kpcn_diffuse_in':torch.ones((8, 34, 128, 128)), 
#         'kpcn_specular_in':torch.ones((8, 34, 128, 128)), 
#         'kpcn_diffuse_buffer':torch.ones((8, 3, 128, 128)),
#         'kpcn_specular_buffer':torch.ones((8, 3, 128, 128)),
#         'kpcn_albedo':torch.ones((8, 3, 128, 128))}
# mask, batch1, batch2 = decomp(data, use_llpm_buf=False)
# print(batch1['kpcn_diffuse_in'].shape)
# print(batch2['target_diffuse'].mean())
# print(batch1['target_specular'])
# feat = FeatureEncoder()
# decomp = decompOriginModule(in_channel=53, out_channel=103)
# data = torch.randn(8, 34, 128, 128)
# enc = feat(data)
# print(enc.shape)
# x = torch.cat((data[:, :3, :, :], enc), dim=1)
# print(x.shape)
# mask, f1, f2 = decomp(torch.cat((data[:, :3, :, :], enc), dim=1))
# print(mask.shape, f1.shape, f2.shape)

# b = torch.randn((8, 34, 128, 128))
# # f = FeatureEncoder()
# # i = f(b)
# # print(i.shape)
# d = decompOriginModule()
# mask, f1, f2 = d(b)
# print(mask.shape, f1.shape, f2.shape)
# b = torch.randn((8, 3, 128, 128))
# v = b.var(2).mean(2, keepdims=True)
# print(v.shape)
# def _gradients(buf):
#         """Compute the xy derivatives of the input buffer. This helper is used in the _preprocess_<base_model>(...) functions
#         Args:
#             buf(np.array)[B, C, H, W]: input image-like tensor.
#         Returns:
#             (np.array)[B, C, H, W]: horizontal and vertical gradients of buf.
#         """
#         dx = buf[:, :, :, 1:] - buf[:, :, :, :-1]
#         dy = buf[:, :, 1:] - buf[:, :, :-1]
#         dx = F.pad(dx, (1,0), "constant", 0.0) # zero padding to the leftni
#         dy = F.pad(dy, (0,0,1,0), 'constant', 0.0)  # zero padding to the up
#         print(dx.shape, dy.shape)
#         return torch.cat([dx, dy], 1)
    
# b = torch.randn((8, 3, 128, 128))
# g = _gradients(b)
# print(g.shape)