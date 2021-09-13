import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import csv
import random
from tqdm import tqdm

from utils import *
from kpcn import *
from kpal import *
from multiscale import *
from decomp import *

from losses import *
from dataset import MSDenoiseDataset, init_data

# from test_cython import *

# L = 9 # number of convolutional layers
# n_kernels = 100 # number of kernels in each layer
# kernel_size = 5 # size of kernel (square)

# # input_channels = dataset[0]['X_diff'].shape[-1]
# hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316

parser = argparse.ArgumentParser(description='Train the model')

'''
Needed parameters
1. Data & Model specifications
device : which device will the data & model should be loaded
mode : which kind of model should it train
input_channel : input channel
hidden_channel : hidden channel
num_layer : number of layers / depth of models
'''
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=34, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

'''
2. Preprocessing specifications
eps
'''
parser.add_argument('--eps', default=0.00316, type=float)

'''
3. Training Specification
val : should it perform validation
early_stopping : should it perform early stopping
trainset : dataset for training
valset : dataset for validation
lr : learning rate
epoch : epoch
criterion : which loss function should it use
'''
parser.set_defaults(do_feature_dropout=False)
parser.add_argument('--do_feature_dropout', dest='do_feature_dropout', action='store_true')
parser.set_defaults(do_finetune=False)
parser.add_argument('--do_finetune', dest='do_finetune', action='store_true')
parser.set_defaults(do_val=False)
parser.add_argument('--do_val', dest='do_val', action='store_true')
parser.set_defaults(do_early_stopping=False)
parser.add_argument('--do_early_stopping', dest='do_early_stopping', action='store_true')
parser.add_argument('--data_dir')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--loss', default='L1')

save_dir = 'decomp_kpcn'
writer = SummaryWriter('tryouts/'+save_dir)

def validation(diffuseNet, specularNet, dataloader, eps, criterion, device, epoch, mode='kpcn'):
  pass
  lossDiff = 0
  lossSpec = 0
  lossFinal = 0
  relL2Final = 0
  relL2 = RelativeMSE()
  # for batch_idx, data in enumerate(dataloader):
  batch_idx = 0
  diffuseNet, specularNet = diffuseNet.eval(), specularNet.eval()
  with torch.no_grad():
    for data in tqdm(dataloader, leave=False, ncols=70):
      X_diff = data['kpcn_diffuse_in'].to(device)
      Y_diff = data['target_diffuse'].to(device)

      # print(diffuseNet.layer)

      # if batch_idx == 10:
      #   pass
      #   diffuseNet.

      outputDiff = diffuseNet(X_diff)
      # if mode == 'KPCN':
      if 'decomp' not in mode and 'kpcn' in mode or 'kpal' in mode:
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)
      lossDiff += criterion(outputDiff, Y_diff).item()

      X_spec = data['kpcn_specular_in'].to(device)
      Y_spec = data['target_specular'].to(device)
      
      outputSpec = specularNet(X_spec)
      # if mode == 'KPCN':
      if 'decomp' not in mode and 'kpcn' in mode or 'kpal' in mode:
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)
      lossSpec += criterion(outputSpec, Y_spec).item()

      # calculate final ground truth error
      albedo = data['kpcn_albedo'].to(device)
      albedo = crop_like(albedo, outputDiff)
      outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

      Y_final = data['target_total'].to(device)
      Y_final = crop_like(Y_final, outputFinal)
      lossFinal += criterion(outputFinal, Y_final).item()
      relL2Final += relL2(outputFinal, Y_final).item()

      # visualize
      if batch_idx == 20:
        inputFinal = data['kpcn_diffuse_buffer'] * (data['kpcn_albedo'] + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0
        inputGrid = torchvision.utils.make_grid(inputFinal)
        writer.add_image('noisy patches e{}'.format(epoch+1), inputGrid)
        writer.add_image('noisy patches e{}'.format(str(epoch+1)+'_'+str(batch_idx)), inputGrid)

        outputGrid = torchvision.utils.make_grid(outputFinal)
        writer.add_image('denoised patches e{}'.format(str(epoch+1)+'_'+str(batch_idx)), outputGrid)
        # writer.add_image('denoised patches e{}'.format(epoch+1), outputGrid)

        cleanGrid = torchvision.utils.make_grid(Y_final)
        # writer.add_image('clean patches e{}'.format(epoch+1), cleanGrid)
        writer.add_image('clean patches e{}'.format(str(epoch+1)+'_'+str(batch_idx)), cleanGrid)



      batch_idx += 1


    return lossDiff/(4*len(dataloader)), lossSpec/(4*len(dataloader)), lossFinal/(4*len(dataloader)), relL2Final/(4*len(dataloader))

def train(mode, device, trainset, validset, eps, L, input_channels, hidden_channels, kernel_size, epochs, learning_rate, loss, do_early_stopping, do_feature_dropout, do_finetune):
  # print('TRAINING WITH VALIDDATASET : {}'.format(validset))
  dataloader = DataLoader(trainset, batch_size=8, num_workers=1, pin_memory=False)
  print(len(dataloader))

  if validset is not None:
    validDataloader = DataLoader(validset, batch_size=4, num_workers=1, pin_memory=False)

  # instantiate networks
  print(L, input_channels, hidden_channels, kernel_size, mode)
  print(mode)
  if mode == 'kpcn':
    diffuseNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  elif mode == 'single_kpal':
    diffuseNet = singleFrameDenoiser(input_channels, hidden_channels, kernel_size=21).to(device)
    specularNet = singleFrameDenoiser(input_channels, hidden_channels, kernel_size=21).to(device)
  elif mode == 'multi_kpcn':
    diffDenoisers, specDenoisers = [], []
    for i in range(3):
       diffDenoisers.append(make_net(L, input_channels, hidden_channels, kernel_size, mode))
       specDenoisers.append(make_net(L, input_channels, hidden_channels, kernel_size, mode))
    diffuseNet = multiScaleDenoiser(hidden_channels, diffDenoisers).to(device)
    specularNet = multiScaleDenoiser(hidden_channels, specDenoisers).to(device)
    diffuseNet._load_denoiser('trained_model/kpcn_finetune_2/diff_e{}.pt'.format(4))
    specularNet._load_denoiser('trained_model/kpcn_finetune_2/spec_e{}.pt'.format(4))
  elif mode == 'decomp_kpcn':
    diffuseNet = decompDenoiser(L, input_channels, hidden_channels, kernel_size, mode, 'trained_model/kpcn_finetune_2/diff_e{}.pt'.format(4)).to(device)
    specularNet = decompDenoiser(L, input_channels, hidden_channels, kernel_size, mode, 'trained_model/kpcn_finetune_2/spec_e{}.pt'.format(4)).to(device)

  print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
  print('# Parameter for diffuseNet : {}'.format(sum([p.numel() for p in diffuseNet.parameters()])))
  print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)
  print('# Parameter for specularNet : {}'.format(sum([p.numel() for p in diffuseNet.parameters()])))

  if loss == 'L1':
    criterion = nn.L1Loss()
  elif loss =='SMAPE':
    criterion = SMAPE()
  else:
    print('Loss Not Supported')
    return
  print('LEARNING RATE : {}'.format(learning_rate))
  optimizerDiff = optim.Adam(diffuseNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
  optimizerSpec = optim.Adam(specularNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))

  # checkpointDiff = torch.load('trained_model/patch_256_kpcn_3/diff_e3.pt')
  # diffuseNet.load_state_dict(checkpointDiff['model_state_dict'])
  # optimizerDiff.load_state_dict(checkpointDiff['optimizer_state_dict'])
  # diffuseNet.load_state_dict(torch.load('trained_model/simple_feat_kpcn_2/diff_e8.pt'))
  diffuseNet.train()

  # checkpointSpec = torch.load('trained_model/patch_256_kpcn_3/spec_e3.pt')
  # specularNet.load_state_dict(checkpointSpec['model_state_dict'])
  # optimizerSpec.load_state_dict(checkpointSpec['optimizer_state_dict'])
  # specularNet.load_state_dict(torch.load('trained_model/simple_feat_kpcn_2/spec_e8.pt'))
  specularNet.train()


  accuLossDiff = 0
  accuLossSpec = 0
  accuLossFinal = 0
  
  lDiff = []
  lSpec = []
  lFinal = []
  valLDiff = []
  valLSpec = []
  valLFinal = []

  # writer = SummaryWriter('runs/'+mode+'_2')
  total_epoch = 0
  # epoch = checkpointDiff['epoch']
  epoch = 0
  # print('Check Initialization')
  # initLossDiff, initLossSpec, initLossFinal, relL2LossFinal = validation(diffuseNet, specularNet, validDataloader, eps, criterion, device, -1, mode)
  # print("initLossDiff: {}".format(initLossDiff))
  # print("initLossSpec: {}".format(initLossSpec))
  # print("initLossFinal: {}".format(initLossFinal))
  # print("relL2LossFinal: {}".format(relL2LossFinal))
  # writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 0, (epoch + 1) * len(validDataloader))
  # writer.add_scalar('Valid total loss', initLossFinal if initLossFinal != float('inf') else 0, (epoch + 1) * len(validDataloader))
  # writer.add_scalar('Valid diffuse loss', initLossDiff if initLossDiff != float('inf') else 0, (epoch + 1) * len(validDataloader))
  # writer.add_scalar('Valid specular loss', initLossSpec if initLossSpec != float('inf') else 0, (epoch + 1) * len(validDataloader))


  import time

  start = time.time()
  print('START')

  # print(len(dataloader))
  # for sample_batched in tqdm(dataloader, leave=False, ncols=70):
  #     # i_batch += 1
  #     print(sample_batched.keys())
  #     print('KPCN_DIFFUSE_IN : {}'.format(sample_batched['kpcn_diffuse_in'].shape))
  #     print('KPCN_DIFFUSE_BUFFER : {}'.format(sample_batched['kpcn_diffuse_buffer'].shape))

  for epoch in range(0, epochs):
    print('EPOCH {}'.format(epoch+1))
    diffuseNet.train()
    specularNet.train()
    # for i_batch, sample_batched in enumerate(dataloader):
    i_batch = -1
    for sample_batched in tqdm(dataloader, leave=False, ncols=70):
      i_batch += 1
      # print(sample_batched.keys())

      # get the inputs
      X_diff = sample_batched['kpcn_diffuse_in'].to(device)
      # X_diff = feature_dropout(X_diff, 1.0, device)

      Y_diff = sample_batched['target_diffuse'].to(device)
      # Y_diff = feature_dropout(Y_diff, 1.0, device)
      # print(X_diff.shape, Y_diff.shape)
      # zero the parameter gradients
      optimizerDiff.zero_grad()

      # forward + backward + optimize
      outputDiff = diffuseNet(X_diff)

      # if mode == 'KPCN':
      if 'decomp' not in mode and 'kpcn' in mode or 'kpal' in mode:
        # print('Outputdiff: ', outputDiff.shape)
        X_input = crop_like(X_diff, outputDiff)
        # print('X_input: ', X_input.shape)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)
      # print('DIFF SHAPES : {}, {}'.format(outputDiff.shape, Y_diff.shape))

      # lossDiff = criterion(outputDiff, Y_diff)
      # lossDiff.backward()
      # optimizerDiff.step()

      # get the inputs
      X_spec = sample_batched['kpcn_specular_in'].to(device)
      Y_spec = sample_batched['target_specular'].to(device)

      # zero the parameter gradients
      optimizerSpec.zero_grad()

      # forward + backward + optimize
      outputSpec = specularNet(X_spec)

      # if mode == 'KPCN':
      if 'decomp' not in mode and 'kpcn' in mode or 'kpal' in mode:
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)

      lossDiff = criterion(outputDiff, Y_diff)
      lossSpec = criterion(outputSpec, Y_spec)
      # if epoch >= 0:
      if not do_finetune:
        lossDiff.backward()
        optimizerDiff.step()
        lossSpec.backward()
        optimizerSpec.step()

      # calculate final ground truth error
      # with torch.no_grad():
      # albedo = sample_batched['origAlbedo'].permute(permutation).to(device)
      albedo = sample_batched['kpcn_albedo'].to(device)
      albedo = crop_like(albedo, outputDiff)
      # print('ALBEDO SIZE: {}'.format(sample_batched['kpcn_albedo'].shape))
      outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

      Y_final = sample_batched['target_total'].to(device)

      Y_final = crop_like(Y_final, outputFinal)

      lossFinal = criterion(outputFinal, Y_final)

      if do_finetune:
        # print('FINETUNING')
        lossFinal.backward()
        optimizerDiff.step()
        optimizerSpec.step()
      accuLossFinal += lossFinal.item()

      accuLossDiff += lossDiff.item()
      accuLossSpec += lossSpec.item()

      writer.add_scalar('lossFinal',  lossFinal if lossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossDiffuse', lossDiff if lossDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossSpec', lossSpec if lossSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    
    accuLossDiff, accuLossSpec, accuLossFinal = accuLossDiff/(8*len(dataloader)), accuLossSpec/(8*len(dataloader)), accuLossFinal/(8*len(dataloader))
    writer.add_scalar('Train total loss', accuLossFinal if accuLossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train diffuse loss', accuLossDiff if accuLossDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train specular loss', accuLossSpec if accuLossSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)


    if not os.path.exists('trained_model/' + save_dir):
      os.makedirs('trained_model/' + save_dir)
      print('MAKE DIR {}'.format('trained_model/'+save_dir))

    # torch.save(diffuseNet.state_dict(), 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': diffuseNet.state_dict(),
            'optimizer_state_dict': optimizerDiff.state_dict(),
            }, 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
    # torch.save(specularNet.state_dict(), 'trained_model/' + save_dir + '/spec_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': specularNet.state_dict(),
            'optimizer_state_dict': optimizerSpec.state_dict(),
            }, 'trained_model/'+ save_dir + '/spec_e{}.pt'.format(epoch+1))
    # print('VALIDATION WORKING!')
    validLossDiff, validLossSpec, validLossFinal, relL2LossFinal = validation(diffuseNet, specularNet, validDataloader, eps, criterion, device, epoch, mode)
    writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid total loss', validLossFinal if accuLossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid diffuse loss', validLossDiff if accuLossDiff != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid specular loss', validLossSpec if accuLossSpec != float('inf') else 1e+35, (epoch + 1) * len(dataloader))

    print("Epoch {}".format(epoch + 1))
    print("LossDiff: {}".format(accuLossDiff))
    print("LossSpec: {}".format(accuLossSpec))
    print("LossFinal: {}".format(accuLossFinal))
    print("ValidrelL2LossDiff: {}".format(relL2LossFinal))
    print("ValidLossDiff: {}".format(validLossDiff))
    print("ValidLossSpec: {}".format(validLossSpec))
    print("ValidLossFinal: {}".format(validLossFinal))

    lDiff.append(accuLossDiff)
    lSpec.append(accuLossSpec)
    lFinal.append(accuLossFinal)
    valLDiff.append(validLossDiff)
    valLSpec.append(validLossSpec)
    valLFinal.append(validLossFinal)

    # if not os.path.exists('trained_model/' + save_dir):
    #   os.makedirs('trained_model/' + save_dir)
    #   print('MAKE DIR {}'.format('trained_model/'+save_dir))

    # # torch.save(diffuseNet.state_dict(), 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': diffuseNet.state_dict(),
    #         'optimizer_state_dict': optimizerDiff.state_dict(),
    #         }, 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
    # # torch.save(specularNet.state_dict(), 'trained_model/' + save_dir + '/spec_e{}.pt'.format(epoch+1))
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': specularNet.state_dict(),
    #         'optimizer_state_dict': optimizerSpec.state_dict(),
    #         }, 'trained_model/'+ save_dir + '/spec_e{}.pt'.format(epoch+1))

    print('SAVED {}/diff_e{}, {}/spec_e{}'.format(save_dir, epoch+1, save_dir, epoch+1))

    total_epoch += 1
    if do_early_stopping and len(valLFinal) > 10 and valLFinal[-1] >= valLFinal[-2]:
      print('EARLY STOPPING!')
      break
    
    accuLossDiff = 0
    accuLossSpec = 0
    accuLossFinal = 0

  writer.close()
  print('Finished training in mode, {} with epoch {}'.format(mode, total_epoch))
  print('Took', time.time() - start, 'seconds.')
  
  return diffuseNet, specularNet, lDiff, lSpec, lFinal


def main():
  args = parser.parse_args()
  print(args)

  dataset, dataloader = init_data(args)
  print(len(dataset['train']), len(dataloader['train']))
  # trainset, validset = dataloader['train'], dataloader['val']
  trainset, validset = dataset['train'], dataset['val']
  print(trainset, validset)

  input_channels = dataset['train'].dncnn_in_size

  train(
    args.mode, 
    args.device, 
    trainset, 
    validset, 
    eps, 
    args.num_layers, 
    input_channels, 
    args.hidden_channels, 
    args.kernel_size, 
    args.epochs, 
    args.lr, 
    args.loss, 
    args.do_early_stopping, 
    args.do_feature_dropout,
    args.do_finetune)
  


if __name__ == '__main__':
  main()