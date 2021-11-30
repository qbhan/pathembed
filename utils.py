import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np
import os

def apply_mask(input, mask):
    i1 = input[:, :3, :, :] * mask
    i1_var = i1.var(1).unsqueeze(1)
    return

def apply_kernel(weights, data):
    # print('WEIGHTS: {}, DATA : {}'.format(weights.shape, data.shape))
    # apply softmax to kernel weights
    # print(weights.shape)
    recon_kernel_size = int(weights.shape[1]**0.5)
    # print()
    weights = weights.permute((0, 2, 3, 1))
    # print(weights.shape, data.shape)
    _, _, h, w = data.size()
    weights = F.softmax(weights, dim=3).view(-1, w * h, recon_kernel_size, recon_kernel_size)
    # print(weights.shape, data.shape)
    # now we have to apply kernels to every pixel
    # first pad the input
    r = recon_kernel_size // 2
    data = F.pad(data[:,:3,:,:], (r,) * 4, "reflect")
    # print(data.shape)
    #print(data[0,:,:,:])
    
    # make slices
    R = []
    G = []
    B = []
    kernels = []
    for i in range(h):
      for j in range(w):
        pos = i*h+j
        # ws = weights[:,pos:pos+1,:,:]
        # kernels += [ws, ws, ws]
        sy, ey = i+r-r, i+r+r+1
        sx, ex = j+r-r, j+r+r+1
        R.append(data[:,0:1,sy:ey,sx:ex])
        G.append(data[:,1:2,sy:ey,sx:ex])
        B.append(data[:,2:3,sy:ey,sx:ex])
        #slices.append(data[:,:,sy:ey,sx:ex])
        
    reds = (torch.cat(R, dim=1)*weights).sum(2).sum(2)
    greens = (torch.cat(G, dim=1)*weights).sum(2).sum(2)
    blues = (torch.cat(B, dim=1)*weights).sum(2).sum(2)
    
    res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w)
    # print(res.shape)
    
    return res

def to_torch_tensors(data):
  if isinstance(data, dict):
    for k, v in data.items():
      if not isinstance(v, torch.Tensor):
        data[k] = torch.from_numpy(v)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if not isinstance(v, torch.Tensor):
        data[i] = to_torch_tensors(v)
    
  return data

# class Scatter2Gather(th.autograd.Function):
#     """Converts (transposes) scatter kernels into gather kernels.
#     Kernel weights at (x, y) for offset (dx, dy) (i.e. scatter[., dy, dx, y,
#     x]) are put at gather[., -dy, -dx, y+dy, x+dx].
#     Args:
#       data(th.Tensor)[bs, k_h, k_w, h, w]: scatter kernel weights.
#     Returns:
#       (th.Tensor)[bs, k_h, k_w, h, w]: gather kernel weights.
#     """
#     @staticmethod
#     def forward(ctx, data):
#         output = data.new()
#         output.resize_as_(data)
#         assert len(data.shape) == 5, "data should be 5d"
#         if _is_cuda(data):
#             ops.scatter2gather_cuda_float32(data, output)
#         else:
#             ops.scatter2gather_cpu_float32(data, output)
#         return output

#     @staticmethod
#     def backward(ctx, d_output):
#         d_data = d_output.new()
#         d_data.resize_as_(d_output)
#         _, kh, kw, _, _ = d_data.shape
#         if _is_cuda(d_output):
#             ops.scatter2gather_cuda_float32(d_output, d_data)
#         else:
#             ops.scatter2gather_cpu_float32(d_output, d_data)
#         return d_data
      


def send_to_device(data, device):
  if isinstance(data, dict):
    for k, v in data.items():
      if isinstance(v, torch.Tensor):
        data[k] = v.to(device)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if isinstance(v, torch.Tensor):
        data[i] = v.to(device)
    
  return data


def getsize(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object reffered to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

def plot_training(diff, spec, filename):
  pass
  plt.plot(diff, 'r', label="Diffuse")
  plt.plot(spec, 'b', label="Specular")
  plt.title(filename)
  plt.xlim([0, 40])
  plt.xlabel('Loss')
  plt.ylabel('Epoch')
  plt.legend()
  plt.savefig(filename + '.jpg')
  # plt.show()


def ToneMap(c, limit=1.5):
    # c: (W, H, C=3)
    luminance = 0.2126 * c[:,:,0] + 0.7152 * c[:,:,1] + 0.0722 * c[:,:,2]
    col = c.copy()
    col[:,:,0] /=  (1.0 + luminance / limit)
    col[:,:,1] /=  (1.0 + luminance / limit)
    col[:,:,2] /=  (1.0 + luminance / limit)
    return col


def ToneMapTest(c, limit=1.5):
  # c: (C=3, W, H)
  luminance = 0.2126 * c[0,:,:] + 0.7152 * c[1,:,:] + 0.0722 * c[2,:,:]
  col = c.clone().detach()
  col[0,:,:] /=  (1.0 + luminance / limit)
  col[1,:,:] /=  (1.0 + luminance / limit)
  col[2,:,:] /=  (1.0 + luminance / limit)
  kInvGamma = 1.0 / 2.2
  return torch.clip(col ** kInvGamma, 0.0, 1.0)

def LinearToSrgb(c):
    # c: (W, H, C=3)
    kInvGamma = 1.0 / 2.2
    return np.clip(c ** kInvGamma, 0.0, 1.0)

def ToneMapBatch(c):
    # c: (B, C=3, W, H)
    luminance = 0.2126 * c[:,0,:,:] + 0.7152 * c[:,1,:,:] + 0.0722 * c[:,2,:,:]
    col = c.clone().detach()
    col[:,0,:,:] /= (1 + luminance / 1.5)
    col[:,1,:,:] /= (1 + luminance / 1.5)
    col[:,2,:,:] /= (1 + luminance / 1.5)
    # col = torch.clip(col,   0, None)
    kInvGamma = 1.0 / 2.2
    # return torch.clip(col ** kInvGamma, 0.0, 1.0)
    return c


def crop_like(src, tgt):
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    # delta = (src_sz[2:4]-tgt_sz[2:4])
    delta = (src_sz[-2:]-tgt_sz[-2:])
    crop = np.maximum(delta // 2, 0)  # no negative crop
    crop2 = delta - crop

    if (crop > 0).any() or (crop2 > 0).any():
        # NOTE: convert to ints to enable static slicing in ONNX conversion
        src_sz = [int(x) for x in src_sz]
        crop = [int(x) for x in crop]
        crop2 = [int(x) for x in crop2]
        return src[..., crop[0]:src_sz[-2]-crop2[0],
                   crop[1]:src_sz[-1]-crop2[1]]
    else:
        return src


def trial_name(args):
  train_dir = 'trained_model'
  trial = args.mode
  if args.lr != 1e-4:
    trial += '_' + str(args.lr)
  if args.loss != 'L1':
    trial += '_' + args.loss

  # add more values

  # print(trial)

  # check if the same trial exists
  i = 1
  trial += '_' + str(i)
  while os.path.exists(os.path.join(train_dir, trial)):
    i += 1
    trial = trial[:-1] + str(i)

  # print(os.path.join(train_dir, trial))
  return trial


def albedo_process(albedo, rad):
  background_mask = albedo[:,:,:,:] == 0
  print(background_mask.shape)