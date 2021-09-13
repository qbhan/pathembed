import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import argparse
import os
from tqdm import tqdm
import csv

from utils import *
from kpcn import *
from dataset import DenoiseDataset
from losses import *

# from test_cython import *

# L = 9 # number of convolutional layers
# n_kernels = 100 # number of kernels in each layer
# kernel_size = 5 # size of kernel (square)

# # input_channels = dataset[0]['X_diff'].shape[-1]
# hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316

parser = argparse.ArgumentParser(description='Test the model')


parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=34, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

parser.add_argument('--diffuse_model')
parser.add_argument('--specular_model')

parser.add_argument('--data_dir')
parser.add_argument('--save_dir')

parser.set_defaults(do_vis_feat=False)
parser.add_argument('--do_vis_feat', dest='do_vis_feat', action='store_true')
# parser.set_defaults(do_errormap=False)
# parser.add_argument('--do_errormap', dest='do_errormap', action='store_true')

save_dir = 'kpcn_k5'
writer = SummaryWriter('test_runs/'+save_dir)

def unsqueeze_all(d):
  for k, v in d.items():
    d[k] = torch.unsqueeze(v, dim=0)
  return d


def denoise(diffuseNet, specularNet, dataloader, device, mode, save_dir, do_vis_feat, debug=False):
  print(len(dataloader))
  with torch.no_grad():
    criterion = nn.L1Loss()
    relL2 = RelativeMSE()
    lossDiff, lossSpec, lossFinal, relL2Final= 0,0,0,0
    scenelossDiff, scenelossSpec, scenelossFinal, scenerelL2Final= 0,0,0,0
    image_idx = 0
    input_image = torch.zeros((3, 960, 960)).to(device)
    gt_image = torch.zeros((3, 960, 960)).to(device)
    output_image = torch.zeros((3, 960, 960)).to(device)

    # Auxiliary features
    if do_vis_feat:
      diff_rad = torch.zeros((3, 960, 960)).to(device)
      diff_rad_var = torch.zeros((1, 960, 960)).to(device)
      diff_rad_dx = torch.zeros((3, 960, 960)).to(device)
      diff_rad_dy = torch.zeros((3, 960, 960)).to(device)
      spec_rad = torch.zeros((3, 960, 960)).to(device)
      spec_rad_var = torch.zeros((1, 960, 960)).to(device)
      spec_rad_dx = torch.zeros((3, 960, 960)).to(device)
      spec_rad_dy = torch.zeros((3, 960, 960)).to(device)
      normal = torch.zeros((3, 960, 960)).to(device)
      normal_var = torch.zeros((1, 960, 960)).to(device)
      normal_dx = torch.zeros((3, 960, 960)).to(device)
      normal_dy = torch.zeros((3, 960, 960)).to(device)
      depth = torch.zeros((1, 960, 960)).to(device)
      depth_var = torch.zeros((1, 960, 960)).to(device)
      depth_dx = torch.zeros((1, 960, 960)).to(device)
      depth_dy = torch.zeros((1, 960, 960)).to(device)
      albedo_in = torch.zeros((3, 960, 960)).to(device)
      albedo_in_var = torch.zeros((1, 960, 960)).to(device)
      albedo_in_dx = torch.zeros((3, 960, 960)).to(device)
      albedo_in_dy = torch.zeros((3, 960, 960)).to(device)

    # Error calculation
    error_map = torch.zeros((3, 960, 960)).to(device)

    x, y = 0, 0
    for data in tqdm(dataloader, leave=False, ncols=70):
      # print(x, y)
      X_diff = data['kpcn_diffuse_in'].to(device)
      Y_diff = data['target_diffuse'].to(device)

      outputDiff = diffuseNet(X_diff)
      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)
      lossDiff += criterion(outputDiff, Y_diff).item()
      scenelossDiff += criterion(outputDiff, Y_diff).item()

      X_spec = data['kpcn_specular_in'].to(device)
      Y_spec = data['target_specular'].to(device)
      
      outputSpec = specularNet(X_spec)
      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)
      lossSpec += criterion(outputSpec, Y_spec).item()
      scenelossSpec += criterion(outputSpec, Y_spec).item()

      # calculate final ground truth error
      albedo = data['kpcn_albedo'].to(device)
      albedo = crop_like(albedo, outputDiff)
      outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

      Y_final = data['target_total'].to(device)
      Y_final = crop_like(Y_final, outputFinal)
      # print(lossFinal, relL2Final)
      lossFinal += criterion(outputFinal, Y_final).item()
      scenelossFinal += criterion(outputFinal, Y_final).item()
      relL2Final += relL2(outputFinal, Y_final).item()
      scenerelL2Final += relL2(outputFinal, Y_final).item()


      # visualize
      inputFinal = data['kpcn_diffuse_buffer'] * (data['kpcn_albedo'] + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0


      # print(np.shape(inputFinal))
      # print(np.shape(outputFinal))
      # print(np.shape(Y_final))
      input_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(inputFinal[0, :, 32:96, 32:96])
      output_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(outputFinal[0, :, 16:80, 16:80])
      gt_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(Y_final[0, :, 16:80, 16:80])
      error_map[:, x*64:x*64+64, y*64:y*64+64] = torch.abs(gt_image[:, x*64:x*64+64, y*64:y*64+64] - output_image[:, x*64:x*64+64, y*64:y*64+64])


      if do_vis_feat:
        diff_rad[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,:3,:,:][0, :3, 32:96, 32:96]
        diff_rad_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,3,:,:][0, 32:96, 32:96]
        diff_rad_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,4:7,:,:][0, :, 32:96, 32:96]
        diff_rad_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,7:10,:,:][0, :, 32:96, 32:96]
        spec_rad[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,:3,:,:][0, :3, 32:96, 32:96]
        spec_rad_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,3,:,:][0, 32:96, 32:96]
        spec_rad_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,4:7,:,:][0, :, 32:96, 32:96]
        spec_rad_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,7:10,:,:][0, :, 32:96, 32:96]
        normal[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,10:13,:,:][0, :, 32:96, 32:96]
        normal_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,13,:,:][0, 32:96, 32:96]
        normal_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,14:17,:,:][0, :, 32:96, 32:96]
        normal_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,17:20,:,:][0, :, 32:96, 32:96]
        depth[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,20,:,:][0, 32:96, 32:96]
        depth_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,21,:,:][0, 32:96, 32:96]
        depth_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,22,:,:][0, 32:96, 32:96]
        depth_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,23,:,:][0, 32:96, 32:96]
        albedo_in[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,24:27,:,:][0, :, 32:96, 32:96]
        albedo_in_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,27,:,:][0, 32:96, 32:96]
        albedo_in_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,28:31,:,:][0, :, 32:96, 32:96]
        albedo_in_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,31:34,:,:][0, :, 32:96, 32:96]

      if 'simple' in mode: 
        if not os.path.exists(save_dir + '/test{}/attns'.format(image_idx)):
            os.makedirs(save_dir + '/test{}/attns'.format(image_idx))
        foutDiff = open(save_dir + '/test{}/attns/patch_{}_diff_attn.csv'.format(image_idx, 15*x+y), 'w')
        foutSpec = open(save_dir + '/test{}/attns/patch_{}_spec_attn.csv'.format(image_idx, 15*x+y), 'w')
        for fi in range(0, 9):
          pass
          fDiff = open('acts/Act_diff_{}.csv'.format(fi))
          fDiff.__next__()
          fSpec = open('acts/Act_spec_{}.csv'.format(fi))
          fSpec.__next__()
          for lineDiff in fDiff:
            # print(line)
            foutDiff.write(str(fi)+ lineDiff[1:])
          for lineSpec in fSpec:
            foutSpec.write(str(fi) + lineSpec[1:])
          fDiff.close()
          fSpec.close()
        foutDiff.close()
        foutSpec.close()

      elif 'spc' in mode:
        if not os.path.exists(save_dir + '/test{}/attns'.format(image_idx)):
            os.makedirs(save_dir + '/test{}/attns'.format(image_idx))
        foutDiff = open(save_dir + '/test{}/attns/patch_{}_diff_attn.csv'.format(image_idx, 15*x+y), 'w')
        foutSpec = open(save_dir + '/test{}/attns/patch_{}_spec_attn.csv'.format(image_idx, 15*x+y), 'w')
        for fi in range(1, 9):
          for fj in range(21):
            fDiff = open('acts/spc/Act_diff_{}_{}.csv'.format(fi, fj))
            fDiff.__next__()
            fSpec = open('acts/spc/Act_spec_{}_{}.csv'.format(fi, fj))
            fSpec.__next__()
            for lineDiff in fDiff:
              # print(line)
              foutDiff.write(str(fi)+ lineDiff[1:])
            for lineSpec in fSpec:
              foutSpec.write(str(fi) + lineSpec[1:])
            fDiff.close()
            fSpec.close()
        foutDiff.close()
        foutSpec.close()


      
      y += 1
      if x < 15 and y>=15:
        x += 1
        y = 0

      if x >= 15:
        if not os.path.exists(save_dir + '/test{}'.format(image_idx)):
          os.makedirs(save_dir + '/test{}'.format(image_idx))
        if not os.path.exists(save_dir + '/test{}/features'.format(image_idx)):
          os.makedirs(save_dir + '/test{}/features'.format(image_idx))
        # if not os.path.exists(save_dir + '/test{}/attns'.format(image_idx)):
        #   os.makedirs(save_dir + '/test{}/attns'.format(image_idx))

        save_image(input_image, save_dir + '/test{}/noisy.png'.format(image_idx))
        save_image(output_image, save_dir + '/test{}/denoise.png'.format(image_idx))
        save_image(gt_image, save_dir + '/test{}/clean.png'.format(image_idx))
        save_image(error_map, save_dir + '/test{}/error_map.png'.format(image_idx))

        # losses
        scenelossDiff, scenelossSpec, scenelossFinal, scenerelL2Final = scenelossDiff/225, scenelossSpec/225, scenelossFinal/225, scenerelL2Final/225
        print()
        print('test {} lossDiff : {}'.format(image_idx, scenelossDiff))
        print('test {} lossSpec : {}'.format(image_idx, scenelossSpec))
        print('test {} lossFinal : {}'.format(image_idx, scenelossFinal))
        print('test {} relL2Final : {}'.format(image_idx, scenerelL2Final))
        writer.add_scalar('test lossDiff',  scenelossDiff if lossFinal != float('inf') else 1e+35, image_idx)
        writer.add_scalar('test lossSpec',  scenelossSpec if lossFinal != float('inf') else 1e+35, image_idx)
        writer.add_scalar('test lossFinal',  scenelossFinal if lossFinal != float('inf') else 1e+35, image_idx)
        writer.add_scalar('test relL2Final',  scenerelL2Final if lossFinal != float('inf') else 1e+35, image_idx)

        if do_vis_feat:
          save_image(diff_rad, save_dir + '/test{}/features/diff_rad.png'.format(image_idx))
          save_image(diff_rad_var, save_dir + '/test{}/features/diff_rad_var.png'.format(image_idx))
          save_image(diff_rad_dx, save_dir + '/test{}/features/diff_rad_dx.png'.format(image_idx))
          save_image(diff_rad_dy, save_dir + '/test{}/features/diff_rad_dy.png'.format(image_idx))
          save_image(spec_rad, save_dir + '/test{}/features/spec_rad.png'.format(image_idx))
          save_image(spec_rad_var, save_dir + '/test{}/features/spec_rad_var.png'.format(image_idx))
          save_image(spec_rad_dx, save_dir + '/test{}/features/spec_rad_dx.png'.format(image_idx))
          save_image(spec_rad_dy, save_dir + '/test{}/features/spec_rad_dy.png'.format(image_idx))
          save_image(normal, save_dir + '/test{}/features/normal.png'.format(image_idx))
          save_image(normal_var, save_dir + '/test{}/features/normal_var.png'.format(image_idx))
          save_image(normal_dx, save_dir + '/test{}/features/normal_dx.png'.format(image_idx))
          save_image(normal_dy, save_dir + '/test{}/features/normal_dy.png'.format(image_idx))
          save_image(depth, save_dir + '/test{}/features/depth.png'.format(image_idx))
          save_image(depth_var, save_dir + '/test{}/features/depth_var.png'.format(image_idx))
          save_image(depth_dx, save_dir + '/test{}/features/depth_dx.png'.format(image_idx))
          save_image(depth_dy, save_dir + '/test{}/features/depth_dy.png'.format(image_idx))
          save_image(albedo_in, save_dir + '/test{}/features/albedo.png'.format(image_idx))
          save_image(albedo_in_var, save_dir + '/test{}/features/albedo_var.png'.format(image_idx))
          save_image(albedo_in_dx, save_dir + '/test{}/features/albedo_dx.png'.format(image_idx))
          save_image(albedo_in_dy, save_dir + '/test{}/features/albedo_dy.png'.format(image_idx))
          
        
        # print('SAVED IMAGES')
        # init
        x, y = 0, 0
        scenelossDiff, scenelossSpec, scenelossFinal, scenerelL2Final = 0, 0, 0, 0
        image_idx += 1


  return lossDiff/len(dataloader), lossSpec/len(dataloader), lossFinal/len(dataloader), relL2Final/len(dataloader)

def test_model(diffuseNet, specularNet, device, data_dir, mode, args):
  pass
  diffuseNet.to(device)
  specularNet.to(device)
  dataset = DenoiseDataset(data_dir, 8, 'kpcn', 'test', 1, 'recon',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
  dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
  _, _, totalL1, totalrelL2 = denoise(diffuseNet, specularNet, dataloader, device, mode, args.save_dir, args.do_vis_feat)
  print('TEST L1 LOSS is {}'.format(totalL1))
  print('TEST L2 LOSS is {}'.format(totalrelL2))


def main():
  args = parser.parse_args()
  print(args)

  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

    # diffuseNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    # specularNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    diffuseNet = KPCN(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size)
    specularNet = KPCN(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size)

  # print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
  # print(torch.load(args.diffuse_model).keys)
  checkpointDiff = torch.load(args.diffuse_model)
  checkpointSpec = torch.load(args.specular_model)
  diffuseNet.load_state_dict(checkpointDiff['model_state_dict'])
  specularNet.load_state_dict(checkpointSpec['model_state_dict'])
  test_model(diffuseNet, specularNet, args.device, args.data_dir, args.mode, args)



if __name__ == '__main__':
  main()