"""
Testing hard-coded decomposition of images
Image will decompose with a certain threshold according to the error map
How about that?
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from torchsummary import summary

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
from path import *

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
parser.set_defaults(do_discrete=False)
parser.add_argument('--do_discrete', dest='do_discrete', action='store_true')

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
parser.add_argument('--use_llpm_buf', default=False, type=bool)
parser.set_defaults(do_val=False)
parser.add_argument('--do_val', dest='do_val', action='store_true')
parser.set_defaults(do_early_stopping=False)
parser.add_argument('--do_early_stopping', dest='do_early_stopping', action='store_true')
parser.add_argument('--data_dir')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--loss', default='L1')

save_dir = 'kpcn_decomp_c1'
writer = SummaryWriter('kpcn/'+save_dir)


def validation(models, dataloader, eps, criterion, device, epoch, use_llpm_buf, mode='kpcn'):
    pass
    lossDiff1 = 0
    lossSpec1 = 0
    lossDiff2 = 0
    lossSpec2 = 0
    lossFinal = 0
    relL2Final = 0
    lossDiffPath = 0
    lossSpecPath = 0
    relL2 = RelativeMSE()
    path_criterion = GlobalRelativeSimilarityLoss()
    # for batch_idx, data in enumerate(dataloader):
    batch_idx = 0
    decompNet = models['decomp']
    diffuseNet1, specularNet1, diffuseNet2, specularNet2 = models['diffuse1'].eval(), models['specular1'].eval(), models['diffuse2'].eval(), models['specular2'].eval()
    diffPathNet, specPathNet = models['path_diffuse'].eval(), models['path_specular'].eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, ncols=70):
             
            # Decompose image
            # print(batch['kpcn_specular_in'].shape)
            for k, v in batch.items():
                batch[k] = v.to(device)
            mask, batch1, batch2 = decompNet(batch)
            

            # if use_llpm_buf:
            paths = batch2['paths'].to(device)
            p_buffer_diffuse, p_buffer_specular = diffPathNet(paths), specPathNet(paths)
            '''Feature Disentanglement'''    
            #TODO
            _, _, c, _, _ = p_buffer_diffuse.shape
            assert c >= 2
            
            # Variance
            p_var_diffuse = p_buffer_diffuse.var(1).mean(1, keepdims=True)
            p_var_diffuse /= p_buffer_diffuse.shape[1]
            p_var_specular = p_buffer_specular.var(1).mean(1, keepdims=True)
            p_var_specular /= p_buffer_specular.shape[1]

            # make new batch
            batch2 = {
                'target_total': batch2['target_total'].to(device),
                'target_diffuse': batch2['target_diffuse'].to(device),
                'target_specular': batch2['target_specular'].to(device),
                'kpcn_diffuse_in': torch.cat([batch2['kpcn_diffuse_in'].to(device), p_buffer_diffuse.mean(1), p_var_diffuse], 1),
                'kpcn_specular_in': torch.cat([batch2['kpcn_specular_in'].to(device), p_buffer_specular.mean(1), p_var_specular], 1),
                'kpcn_diffuse_buffer': batch2['kpcn_diffuse_buffer'].to(device),
                'kpcn_specular_buffer': batch2['kpcn_specular_buffer'].to(device),
                'kpcn_albedo': batch2['kpcn_albedo'].to(device),
            }

            # Denosing using only G-buffers

            # inputs
            X_diff1 = batch1['kpcn_diffuse_in'].to(device)
            Y_diff1 = batch1['target_diffuse'].to(device)
            X_spec1 = batch1['kpcn_specular_in'].to(device)
            Y_spec1 = batch1['target_specular'].to(device)

            outputDiff1 = diffuseNet1(X_diff1)
            Y_diff1 = crop_like(Y_diff1, outputDiff1)
            lossDiff1 += criterion(outputDiff1, Y_diff1).item()

            outputSpec1 = specularNet1(X_spec1)
            Y_spec1 = crop_like(Y_spec1, outputSpec1)
            lossSpec1 += criterion(outputSpec1, Y_spec1).item()

            # calculate final ground truth error
            albedo = batch1['kpcn_albedo'].to(device)
            albedo = crop_like(albedo, outputDiff1)
            outputFinal1 = outputDiff1 * (albedo + eps) + torch.exp(outputSpec1) - 1.0

            
            # Denoising using G-buffers & P-buffers

            # inputs
            X_diff2 = batch2['kpcn_diffuse_in'].to(device)
            Y_diff2 = batch2['target_diffuse'].to(device)
            X_spec2 = batch2['kpcn_specular_in'].to(device)
            Y_spec2 = batch2['target_specular'].to(device)

            outputDiff2 = diffuseNet2(X_diff2)
            Y_diff2 = crop_like(Y_diff2, outputDiff2)
            lossDiff2 += criterion(outputDiff2, Y_diff2).item()

            outputSpec2 = specularNet2(X_spec2)
            Y_spec2 = crop_like(Y_spec2, outputSpec2)
            lossSpec2 += criterion(outputSpec2, Y_spec2).item()

            # calculate final ground truth error
            albedo = batch2['kpcn_albedo'].to(device)
            albedo = crop_like(albedo, outputDiff2)
            outputFinal2 = outputDiff2 * (albedo + eps) + torch.exp(outputSpec2) - 1.0


            # Loss of merged denoised result
            outputFinal = outputFinal1 + outputFinal2
            Y_final = batch['target_total'].to(device)
            Y_final = crop_like(Y_final, outputFinal)
            lossFinal += criterion(outputFinal, Y_final).item()
            relL2Final += relL2(outputFinal, Y_final).item()

            # if use_llpm_buf:
            p_buffer_diffuse = crop_like(p_buffer_diffuse, outputDiff2)
            loss_manif_diffuse = path_criterion(p_buffer_diffuse, Y_diff2)
            p_buffer_specular = crop_like(p_buffer_specular, outputSpec2)
            loss_manif_specular = path_criterion(p_buffer_specular, Y_spec2)
            lossDiffPath += loss_manif_diffuse
            lossSpecPath += loss_manif_specular
                # lossDiff += 0.1 * loss_manif_diffuse
                # lossSpec += 0.1 * loss_manif_specular

            # visualize
            if batch_idx == 20:
                inputFinal = batch['kpcn_diffuse_buffer'] * (batch['kpcn_albedo'] + eps) + torch.exp(batch['kpcn_specular_buffer']) - 1.0
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

    return lossDiff1/(4*len(dataloader)), lossSpec1/(4*len(dataloader)), lossDiff2/(4*len(dataloader)), lossSpec2/(4*len(dataloader)), lossFinal/(4*len(dataloader)), relL2Final/(4*len(dataloader)), lossDiffPath/(4*len(dataloader)), lossSpecPath/(4*len(dataloader))


def train(mode, 
        device, 
        trainset, 
        validset, 
        eps, 
        L, 
        input_channels, 
        hidden_channels, 
        kernel_size, 
        epochs, 
        learning_rate, 
        loss, 
        do_early_stopping, 
        do_finetune,
        use_llpm_buf,
        do_discrete
        ):
    dataloader = DataLoader(trainset, batch_size=8, num_workers=1, pin_memory=False)
    print(len(dataloader))

    if validset is not None:
        validDataloader = DataLoader(validset, batch_size=4, num_workers=1, pin_memory=False)

    # instantiate networks
    print(L, input_channels, hidden_channels, kernel_size, mode)
    print(mode)
    # decompNet = decompModule(in_channel=26, discrete=do_discrete).to(device)
    # optimizerDecomp = optim.Adam(decompNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    diffuseNet1 = KPCN(L, 34, hidden_channels, kernel_size).to(device)
    specularNet1 = KPCN(L, 34, hidden_channels, kernel_size).to(device)
    diffuseNet2 = KPCN(L, input_channels, hidden_channels, kernel_size).to(device)
    specularNet2 = KPCN(L, input_channels, hidden_channels, kernel_size).to(device)

    print('LEARNING RATE : {}'.format(learning_rate))
    optimizerDiff1 = optim.Adam(diffuseNet1.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    optimizerSpec1 = optim.Adam(specularNet1.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    optimizerDiff2 = optim.Adam(diffuseNet2.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    optimizerSpec2 = optim.Adam(specularNet2.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    diffPathNet = PathNet(trainset.pnet_in_size).to(device)
    optimizerDiffPath = optim.Adam(diffPathNet.parameters(), lr=1e-4, betas=(0.9, 0.99))
    specPathNet = PathNet(trainset.pnet_in_size).to(device)
    optimizerSpecPath = optim.Adam(specPathNet.parameters(), lr=1e-4, betas=(0.9, 0.99))
    path_criterion = GlobalRelativeSimilarityLoss()

    # checkpointDiffPath = torch.load('trained_model/kpcn_decomp_3/path_diff_e5.pt')
    # diffPathNet.load_state_dict(checkpointDiffPath['model_state_dict'])
    # optimizerDiffPath.load_state_dict(checkpointDiffPath['optimizer_state_dict'])
    diffPathNet.train()

    # checkpointSpecPath = torch.load('trained_model/kpcn_decomp_3/path_spec_e5.pt')
    # specPathNet.load_state_dict(checkpointSpecPath['model_state_dict'])
    # optimizerSpecPath.load_state_dict(checkpointSpecPath['optimizer_state_dict'])
    specPathNet.train()
    # else

    print(diffuseNet1, "CUDA:", next(diffuseNet1.parameters()).device)
    print(diffPathNet, "CUDA:", next(diffPathNet.parameters()).device)
    print('# Parameter for KPCN : {}'.format(sum([p.numel() for p in diffuseNet1.parameters()])))
    print('# Parameter for PathNet : {}'.format(sum([p.numel() for p in diffPathNet.parameters()])))
    # print(summary(diffuseNet, input_size=(3, 128, 128)))

    if loss == 'L1':
        criterion = nn.L1Loss()
    elif loss =='SMAPE':
        criterion = SMAPE()
    else:
        print('Loss Not Supported')
        return
    # optimizerP = optim.Adam(specularNet.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # checkpointDiff1 = torch.load('trained_model/kpcn_decomp_3/diff1_e5.pt')
    # diffuseNet1.load_state_dict(checkpointDiff1['model_state_dict'])
    # optimizerDiff1.load_state_dict(checkpointDiff1['optimizer_state_dict'])
    diffuseNet1.train()

    # checkpointSpec1 = torch.load('trained_model/kpcn_decomp_3/spec1_e5.pt')
    # specularNet1.load_state_dict(checkpointSpec1['model_state_dict'])
    # optimizerSpec1.load_state_dict(checkpointSpec1['optimizer_state_dict'])
    specularNet1.train()

    # checkpointDiff2 = torch.load('trained_model/kpcn_decomp_3/diff2_e5.pt')
    # diffuseNet2.load_state_dict(checkpointDiff2['model_state_dict'])
    # optimizerDiff2.load_state_dict(checkpointDiff2['optimizer_state_dict'])
    diffuseNet2.train()

    # checkpointSpec2 = torch.load('trained_model/kpcn_decomp_3/spec2_e5.pt')
    # specularNet2.load_state_dict(checkpointSpec2['model_state_dict'])
    # optimizerSpec2.load_state_dict(checkpointSpec2['optimizer_state_dict'])
    specularNet2.train()



    # pNet.train()

    accuLossDiff1 = 0
    accuLossSpec1 = 0
    accuLossDiff2 = 0
    accuLossSpec2 = 0
    accuLossFinal = 0

    lDiff = []
    lSpec = []
    lFinal = []
    valLDiff = []
    valLSpec = []
    valLFinal = []

    # writer = SummaryWriter('runs/'+mode+'_2')
    total_epoch = 0
    init_epoch = 0
    # init_epoch = checkpointDiff1['epoch'] + 1
    # epoch = 0
    
    if init_epoch == 0:
        print('Check Initialization')
        models = { 
                'diffuse1': diffuseNet1, 
                'specular1': specularNet1, 
                'diffuse2': diffuseNet2, 
                'specular2': specularNet2, 
                'path_diffuse': diffPathNet, 
                'path_specular': specPathNet
                }
        initLossDiff1, initLossSpec1, initLossDiff2, initLossSpec2, initLossFinal, relL2LossFinal, pathDiffLoss, pathSpecLoss = validation(models, validDataloader, eps, criterion, device, -1, use_llpm_buf,mode)
        print("initLossDiff1: {}".format(initLossDiff1))
        print("initLossSpec1: {}".format(initLossSpec1))
        print("initLossFinal: {}".format(initLossFinal))
        print("relL2LossFinal: {}".format(relL2LossFinal))
        print("pathDiffLoss: {}".format(pathDiffLoss))
        print("pathSpecLoss: {}".format(pathSpecLoss))
        writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid total loss', initLossFinal if initLossFinal != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid diffuse loss 1', initLossDiff1 if initLossDiff1 != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid specular loss 1', initLossSpec1 if initLossSpec1 != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid diffuse loss 2', initLossDiff2 if initLossDiff2 != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid specular loss 2', initLossSpec2 if initLossSpec2 != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid path diffuse loss', pathDiffLoss if pathDiffLoss != float('inf') else 0, (init_epoch + 1))
        writer.add_scalar('Valid path specular loss', pathSpecLoss if pathSpecLoss != float('inf') else 0, (init_epoch + 1))


    import time

    start = time.time()
    print('START')

    for epoch in range(init_epoch, epochs):
        print('EPOCH {}'.format(epoch+1))
        # decompNet.train()
        diffuseNet1.train()
        specularNet1.train()
        diffuseNet2.train()
        specularNet2.train()
        diffPathNet.train()
        specPathNet.train()
        i_batch = -1
        for batch in tqdm(dataloader, leave=False, ncols=70):
            i_batch += 1
            # print(batch['kpcn_specular_in'].shape)
            # print('DECOMPOSITION')
            for k, v in batch.items():
                batch[k] = v.to(device)

            loss_manif = None
            paths = batch2['paths'].to(device)
            p_buffer_diffuse, p_buffer_specular = diffPathNet(paths), specPathNet(paths)
            '''Feature Disentanglement'''    
            #TODO
            _, _, c, _, _ = p_buffer_diffuse.shape
            assert c >= 2
            
            # Variance
            p_var_diffuse = p_buffer_diffuse.var(1).mean(1, keepdims=True)
            p_var_diffuse /= p_buffer_diffuse.shape[1]
            p_var_specular = p_buffer_specular.var(1).mean(1, keepdims=True)
            p_var_specular /= p_buffer_specular.shape[1]

            # make new batch
            batch2 = {
                'target_total': batch2['target_total'].to(device),
                'target_diffuse': batch2['target_diffuse'].to(device),
                'target_specular': batch2['target_specular'].to(device),
                'kpcn_diffuse_in': torch.cat([batch2['kpcn_diffuse_in'].to(device), p_buffer_diffuse.mean(1), p_var_diffuse], 1),
                'kpcn_specular_in': torch.cat([batch2['kpcn_specular_in'].to(device), p_buffer_specular.mean(1), p_var_specular], 1),
                'kpcn_diffuse_buffer': batch2['kpcn_diffuse_buffer'].to(device),
                'kpcn_specular_buffer': batch2['kpcn_specular_buffer'].to(device),
                'kpcn_albedo': batch2['kpcn_albedo'].to(device),
            }

            # zero the parameter gradients
            optimizerDiff1.zero_grad()
            optimizerSpec1.zero_grad()
            optimizerDiff2.zero_grad()
            optimizerSpec2.zero_grad()
            optimizerDiffPath.zero_grad()
            optimizerSpecPath.zero_grad()

            # Denosing using only G-buffers

            # inputs
            X_diff1 = batch1['kpcn_diffuse_in'].to(device)
            Y_diff1 = batch1['target_diffuse'].to(device)
            X_spec1 = batch1['kpcn_specular_in'].to(device)
            Y_spec1 = batch1['target_specular'].to(device)

            outputDiff1 = diffuseNet1(X_diff1)
            Y_diff1 = crop_like(Y_diff1, outputDiff1)
            lossDiff1 = criterion(outputDiff1, Y_diff1)

            outputSpec1 = specularNet1(X_spec1)
            Y_spec1 = crop_like(Y_spec1, outputSpec1)
            lossSpec1 = criterion(outputSpec1, Y_spec1)

            # calculate final ground truth error
            albedo = batch1['kpcn_albedo'].to(device)
            albedo = crop_like(albedo, outputDiff1)
            outputFinal1 = outputDiff1 * (albedo + eps) + torch.exp(outputSpec1) - 1.0

            # Denoising using G-buffers & P-buffers

            # inputs
            X_diff2 = batch2['kpcn_diffuse_in'].to(device)
            Y_diff2 = batch2['target_diffuse'].to(device)
            X_spec2 = batch2['kpcn_specular_in'].to(device)
            Y_spec2 = batch2['target_specular'].to(device)

            outputDiff2 = diffuseNet2(X_diff2)
            Y_diff2 = crop_like(Y_diff2, outputDiff2)
            lossDiff2 = criterion(outputDiff2, Y_diff2)

            outputSpec2 = specularNet2(X_spec2)
            Y_spec2 = crop_like(Y_spec2, outputSpec2)
            lossSpec2 = criterion(outputSpec2, Y_spec2)

            # calculate final ground truth error
            albedo = batch2['kpcn_albedo'].to(device)
            albedo = crop_like(albedo, outputDiff2)
            outputFinal2 = outputDiff2 * (albedo + eps) + torch.exp(outputSpec2) - 1.0

            
            p_buffer_diffuse = crop_like(p_buffer_diffuse, outputDiff2)
            loss_manif_diffuse = path_criterion(p_buffer_diffuse, Y_diff2)
            p_buffer_specular = crop_like(p_buffer_specular, outputSpec2)
            loss_manif_specular = path_criterion(p_buffer_specular, Y_spec2)
            lossDiff2 += 0.1 * loss_manif_diffuse
            lossSpec2 += 0.1 * loss_manif_specular
                

            if not do_finetune:
                lossDiff1.backward()
                optimizerDiff1.step()
                lossSpec1.backward()
                optimizerSpec1.step()
                lossDiff2.backward()
                optimizerDiff2.step()
                lossSpec2.backward()
                optimizerSpec2.step()
                optimizerDiffPath.step()
                optimizerSpecPath.step()
                optimizerDecomp.step()

                # Loss of merged denoised result
                with torch.no_grad():
                    outputFinal = outputFinal1 + outputFinal2
                    Y_final = batch['target_total'].to(device)
                    Y_final = crop_like(Y_final, outputFinal)
                    lossFinal = criterion(outputFinal, Y_final)
                    # relL2Final = relL2(outputFinal, Y_final).item()

            if do_finetune:
                # print('FINETUNING')
                outputFinal = outputFinal1 + outputFinal2
                Y_final = batch['target_total'].to(device)
                Y_final = crop_like(Y_final, outputFinal)
                lossFinal = criterion(outputFinal, Y_final)
                lossFinal.backward()
                optimizerDiff1.step()
                optimizerSpec1.step()
                optimizerDiff2.step()
                optimizerSpec2.step()
                optimizerDiffPath.step()
                optimizerSpecPath.step()
            
            accuLossDiff1 += lossDiff1.item()
            accuLossSpec1 += lossSpec1.item()
            accuLossDiff2 += lossDiff2.item()
            accuLossSpec2 += lossSpec2.item()
            accuLossFinal += lossFinal.item()

            writer.add_scalar('lossFinal', lossFinal if lossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossDiff1', lossDiff1 if lossDiff1 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossSpec1', lossSpec1 if lossSpec1 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossDiff2', lossDiff2 if lossDiff2 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossSpec2', lossSpec2 if lossSpec2 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    
            accuLossDiff1, accuLossSpec1, accuLossDiff2, accuLossSpec2, accuLossFinal = accuLossDiff1/(8*len(dataloader)), accuLossSpec1/(8*len(dataloader)), accuLossDiff2/(8*len(dataloader)), accuLossSpec2/(8*len(dataloader)), accuLossFinal/(8*len(dataloader))
            writer.add_scalar('Train total loss', accuLossFinal if accuLossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train diffuse loss 1', accuLossDiff1 if accuLossDiff1 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train specular loss 1', accuLossSpec1 if accuLossSpec1 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train diffuse loss 2', accuLossDiff2 if accuLossDiff2 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train specular loss 2', accuLossSpec2 if accuLossSpec2 != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)


        if not os.path.exists('trained_model/' + save_dir):
            os.makedirs('trained_model/' + save_dir)
            print('MAKE DIR {}'.format('trained_model/'+save_dir))

        torch.save({
                'epoch': epoch,
                'model_state_dict': decompNet.state_dict(),
                'optimizer_state_dict': optimizerDecomp.state_dict(),
                }, 'trained_model/'+ save_dir + '/decomp_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': diffuseNet1.state_dict(),
                'optimizer_state_dict': optimizerDiff1.state_dict(),
                }, 'trained_model/'+ save_dir + '/diff1_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': specularNet1.state_dict(),
                'optimizer_state_dict': optimizerSpec1.state_dict(),
                }, 'trained_model/'+ save_dir + '/spec1_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': diffuseNet2.state_dict(),
                'optimizer_state_dict': optimizerDiff2.state_dict(),
                }, 'trained_model/'+ save_dir + '/diff2_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': specularNet2.state_dict(),
                'optimizer_state_dict': optimizerSpec2.state_dict(),
                }, 'trained_model/'+ save_dir + '/spec2_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': diffPathNet.state_dict(),
                'optimizer_state_dict': optimizerDiffPath.state_dict(),
                }, 'trained_model/'+ save_dir + '/path_diff_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': specPathNet.state_dict(),
                'optimizer_state_dict': optimizerSpecPath.state_dict(),
                }, 'trained_model/'+ save_dir + '/path_spec_e{}.pt'.format(epoch+1))
        # print('VALIDATION WORKING!')
        models = {'decomp': decompNet, 
            'diffuse1': diffuseNet1, 
            'specular1': specularNet1, 
            'diffuse2': diffuseNet2, 
            'specular2': specularNet2, 
            'path_diffuse': diffPathNet, 
            'path_specular': specPathNet
            }
        validLossDiff1, validLossSpec1, validLossDiff2, validLossSpec2, validLossFinal, relL2LossFinal, pathDiffLoss, pathSpecLoss = validation(models, validDataloader, eps, criterion, device, epoch, use_llpm_buf,mode)
        writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 1e+35, (epoch + 1))
        writer.add_scalar('Valid total loss', validLossFinal if accuLossFinal != float('inf') else 1e+35, (epoch + 1))
        writer.add_scalar('Valid diffuse loss 1', validLossDiff1 if validLossDiff1 != float('inf') else 0, (epoch + 1))
        writer.add_scalar('Valid specular loss 1', validLossSpec1 if validLossSpec1 != float('inf') else 0, (epoch + 1))
        writer.add_scalar('Valid diffuse loss 2', validLossDiff2 if validLossDiff2 != float('inf') else 0, (epoch + 1))
        writer.add_scalar('Valid specular loss 2', validLossSpec2 if validLossSpec2 != float('inf') else 0, (epoch + 1))
        writer.add_scalar('Valid path diffuse loss', pathDiffLoss if pathDiffLoss != float('inf') else 0, (epoch + 1))
        writer.add_scalar('Valid path specular loss', pathSpecLoss if pathSpecLoss != float('inf') else 0, (epoch + 1))


        print("Epoch {}".format(epoch + 1))
        print("ValidLossDiff1: {}".format(validLossDiff1))
        print("ValidLossSpec1: {}".format(validLossSpec1))
        print("ValidLossDiff2: {}".format(validLossDiff2))
        print("ValidLossSpec2: {}".format(validLossSpec2))
        print("ValidLossFinal: {}".format(validLossFinal))
        print("ValidrelL2LossDiff: {}".format(relL2LossFinal))
        print("pathDiffLoss: {}".format(pathDiffLoss))
        print("pathSpecLoss: {}".format(pathSpecLoss))
        

        # lDiff.append(accuLossDiff)
        # lSpec.append(accuLossSpec)
        # lFinal.append(accuLossFinal)
        # valLDiff.append(validLossDiff)
        # valLSpec.append(validLossSpec)
        # valLFinal.append(validLossFinal)

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
  
    # return diffuseNet, specularNet, lDiff, lSpec, lFinal


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
        args.do_finetune,
        args.use_llpm_buf,
        args.do_discrete
        )
  

if __name__ == '__main__':
  main()