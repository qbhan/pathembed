import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# for mixed precision
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

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
parser.add_argument('--manif_w', default=0.1, type=float)
parser.add_argument('--loss', default='L1')

save_dir = 'kpcn_manif_pad'
writer = SummaryWriter('kpcn/'+save_dir)


def validation(models, dataloader, eps, criterion, device, epoch, use_llpm_buf, mode='kpcn'):
    pass
    lossDiff = 0
    lossSpec = 0
    lossFinal = 0
    relL2Final = 0
    lossDiffPath = 0
    lossSpecPath = 0
    relL2 = RelativeMSE()
    # path_criterion = GlobalRelativeSimilarityLoss()
    path_criterion = FeatureMSE()
    # for batch_idx, data in enumerate(dataloader):
    batch_idx = 0
    if use_llpm_buf:
        diffPathNet, specPathNet = models['path_diffuse'].eval(), models['path_specular'].eval()
    diffuseNet, specularNet = models['diffuse'].eval(), models['specular'].eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, ncols=70):
            # print(data.keys())
            # assert 'paths' in data
            # print('WORKING WITH PATH')
            # print(batch['kpcn_diffuse_in'].shape)
            if use_llpm_buf:
                paths = batch['paths'].to(device)
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
                batch = {
                    'target_total': batch['target_total'].to(device),
                    'target_diffuse': batch['target_diffuse'].to(device),
                    'target_specular': batch['target_specular'].to(device),
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'].to(device), p_buffer_diffuse.mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'].to(device), p_buffer_specular.mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'].to(device),
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'].to(device),
                    'kpcn_albedo': batch['kpcn_albedo'].to(device),
                }

            X_diff = batch['kpcn_diffuse_in'].to(device)
            Y_diff = batch['target_diffuse'].to(device)


            outputDiff = diffuseNet(X_diff)

            Y_diff = crop_like(Y_diff, outputDiff)
            lossDiff += criterion(outputDiff, Y_diff).item()

            X_spec = batch['kpcn_specular_in'].to(device)
            Y_spec = batch['target_specular'].to(device)
            
            outputSpec = specularNet(X_spec)


            Y_spec = crop_like(Y_spec, outputSpec)
            lossSpec += criterion(outputSpec, Y_spec).item()

            # calculate final ground truth error
            albedo = batch['kpcn_albedo'].to(device)
            albedo = crop_like(albedo, outputDiff)
            outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

            Y_final = batch['target_total'].to(device)
            Y_final = crop_like(Y_final, outputFinal)
            lossFinal += criterion(outputFinal, Y_final).item()
            relL2Final += relL2(outputFinal, Y_final).item()

            if use_llpm_buf:
                p_buffer_diffuse = crop_like(p_buffer_diffuse, outputDiff)
                loss_manif_diffuse = path_criterion(p_buffer_diffuse, Y_diff)
                p_buffer_specular = crop_like(p_buffer_specular, outputSpec)
                loss_manif_specular = path_criterion(p_buffer_specular, Y_spec)
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

    return lossDiff/(4*len(dataloader)), lossSpec/(4*len(dataloader)), lossFinal/(4*len(dataloader)), relL2Final/(4*len(dataloader)), lossDiffPath/(4*len(dataloader)), lossSpecPath/(4*len(dataloader))


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
        manif_w
        ):
    dataloader = DataLoader(trainset, batch_size=8, num_workers=1, pin_memory=False)
    print(len(dataloader))

    if validset is not None:
        validDataloader = DataLoader(validset, batch_size=4, num_workers=1, pin_memory=False)

    # instantiate networks
    print(L, input_channels, hidden_channels, kernel_size, mode)
    print(mode)
    if mode == 'kpcn':
        diffuseNet = KPCN(L, input_channels, hidden_channels, kernel_size).to(device)
        specularNet = KPCN(L, input_channels, hidden_channels, kernel_size).to(device)
    # Path module
    
    if use_llpm_buf:
        diffPathNet = PathNet(trainset.pnet_in_size).to(device)
        optimizerDiffPath = optim.Adam(diffPathNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
        specPathNet = PathNet(trainset.pnet_in_size).to(device)
        optimizerSpecPath = optim.Adam(specPathNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
        path_criterion = GlobalRelativeSimilarityLoss()

        # checkpointDiffPath = torch.load('trained_model/kpcn_manif_feat/path_diff_e13.pt')
        # diffPathNet.load_state_dict(checkpointDiffPath['model_state_dict'])
        # optimizerDiffPath.load_state_dict(checkpointDiffPath['optimizer_state_dict'])
        diffPathNet.train()

        # checkpointSpecPath = torch.load('trained_model/kpcn_manif_feat/path_spec_e13.pt')
        # specPathNet.load_state_dict(checkpointSpecPath['model_state_dict'])
        # optimizerSpecPath.load_state_dict(checkpointSpecPath['optimizer_state_dict'])
        specPathNet.train()
    # else

    print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
    print('# Parameter for diffuseNet : {}'.format(sum([p.numel() for p in diffuseNet.parameters()])))
    print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)
    print('# Parameter for specularNet : {}'.format(sum([p.numel() for p in diffuseNet.parameters()])))
    # print(summary(diffuseNet, input_size=(3, 128, 128)))

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
    # optimizerP = optim.Adam(specularNet.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # checkpointDiff = torch.load('trained_model/kpcn_manif_feat/diff_e13.pt')
    # diffuseNet.load_state_dict(checkpointDiff['model_state_dict'])
    # optimizerDiff.load_state_dict(checkpointDiff['optimizer_state_dict'])
    diffuseNet.train()

    # checkpointSpec = torch.load('trained_model/kpcn_manif_feat/spec_e13.pt')
    # specularNet.load_state_dict(checkpointSpec['model_state_dict'])
    # optimizerSpec.load_state_dict(checkpointSpec['optimizer_state_dict'])
    specularNet.train()
    last_epoch = 0
    # last_epoch = checkpointDiff['epoch'] + 1
    print(last_epoch)


    # scaler = GradScaler()

    accuLossDiff = 0
    accuLossSpec = 0
    accuLossPathDiff = 0
    accuLossPathSpec = 0
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
    if last_epoch == 0:
        epoch = 0

        print('Check Initialization')
        models = {'diffuse': diffuseNet, 'specular': specularNet}
        if use_llpm_buf:
            models['path_diffuse'] = diffPathNet
            models['path_specular'] = specPathNet
            # models.append({'path_diffuse': diffPathNet, 'path_specular': specPathNet})
        initLossDiff, initLossSpec, initLossFinal, relL2LossFinal, pathDiffLoss, pathSpecLoss = validation(models, validDataloader, eps, criterion, device, -1, use_llpm_buf,mode)
        print("initLossDiff: {}".format(initLossDiff))
        print("initLossSpec: {}".format(initLossSpec))
        print("initLossFinal: {}".format(initLossFinal))
        print("relL2LossFinal: {}".format(relL2LossFinal))
        print("pathDiffLoss: {}".format(pathDiffLoss))
        print("pathSpecLoss: {}".format(pathSpecLoss))
        writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid total loss', initLossFinal if initLossFinal != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid diffuse loss', initLossDiff if initLossDiff != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid specular loss', initLossSpec if initLossSpec != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid path diffuse loss', pathDiffLoss if pathDiffLoss != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid path specular loss', pathSpecLoss if pathSpecLoss != float('inf') else 0, (epoch + 1) * len(validDataloader))


    import time

    start = time.time()
    print('START')

    for epoch in range(last_epoch, epochs):
        print('EPOCH {}'.format(epoch+1))
        diffuseNet.train()
        specularNet.train()
        i_batch = -1
        for batch in tqdm(dataloader, leave=False, ncols=70):
            i_batch += 1
            # print(sample_batched.keys())
            # with autocast():
            # zero the parameter gradients
            optimizerDiff.zero_grad()
            optimizerSpec.zero_grad()
            if use_llpm_buf:
                optimizerDiffPath.zero_grad()
                optimizerSpecPath.zero_grad()
            # if use_llpm_buf:
                paths = batch['paths'].to(device)
                diffPathNet.train()
                specPathNet.train()
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
                batch = {
                    'target_total': batch['target_total'].to(device),
                    'target_diffuse': batch['target_diffuse'].to(device),
                    'target_specular': batch['target_specular'].to(device),
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'].to(device), p_buffer_diffuse.mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'].to(device), p_buffer_specular.mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'].to(device),
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'].to(device),
                    'kpcn_albedo': batch['kpcn_albedo'].to(device),
                }

            # get the inputs
            X_diff = batch['kpcn_diffuse_in'].to(device)

            Y_diff = batch['target_diffuse'].to(device)

            outputDiff = diffuseNet(X_diff)

            Y_diff = crop_like(Y_diff, outputDiff)


            # get the inputs
            X_spec = batch['kpcn_specular_in'].to(device)
            Y_spec = batch['target_specular'].to(device)

            

            # forward + backward + optimize
            outputSpec = specularNet(X_spec)

            Y_spec = crop_like(Y_spec, outputSpec)
            lossDiff = criterion(outputDiff, Y_diff)
            lossSpec = criterion(outputSpec, Y_spec)

            # loss
            if use_llpm_buf:
                p_buffer_diffuse = crop_like(p_buffer_diffuse, outputDiff)
                loss_manif_diffuse = path_criterion(p_buffer_diffuse, Y_diff)
                p_buffer_specular = crop_like(p_buffer_specular, outputSpec)
                loss_manif_specular = path_criterion(p_buffer_specular, Y_spec)
                lossDiff += manif_w * loss_manif_diffuse
                lossSpec += manif_w * loss_manif_specular
                accuLossPathDiff += loss_manif_diffuse
                accuLossPathSpec += loss_manif_specular
                
            
            if not do_finetune:
                lossDiff.backward()
                lossSpec.backward()
                # scaler.scale(lossDiff).backward()
                # scaler.scale(lossSpec).backward()
                optimizerDiff.step()
                optimizerSpec.step()
                if use_llpm_buf:
                    optimizerDiffPath.step()
                    optimizerSpecPath.step()

            # calculate final ground truth error
            # if not do_finetune:
                with torch.no_grad():
                    albedo = batch['kpcn_albedo'].to(device)
                    albedo = crop_like(albedo, outputDiff)
                    outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

                    Y_final = batch['target_total'].to(device)

                    Y_final = crop_like(Y_final, outputFinal)

                    lossFinal = criterion(outputFinal, Y_final)
            else:
                albedo = batch['kpcn_albedo'].to(device)
                albedo = crop_like(albedo, outputDiff)
                outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

                Y_final = batch['target_total'].to(device)

                Y_final = crop_like(Y_final, outputFinal)

                lossFinal = criterion(outputFinal, Y_final)
                lossFinal.backward()
                optimizerDiff.step()
                optimizerSpec.step()
                if use_llpm_buf:
                    optimizerDiffPath.step()
                    optimizerSpecPath.step()

            # if do_finetune:
            #     # print('FINETUNING')
            #     lossFinal.backward()
            #     optimizerDiff.step()
            #     optimizerSpec.step()
            accuLossFinal += lossFinal.item()

            accuLossDiff += lossDiff.item()
            accuLossSpec += lossSpec.item()

            writer.add_scalar('lossFinal',  lossFinal if lossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossDiffuse', lossDiff if lossDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossSpec', lossSpec if lossSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    
            accuLossDiff, accuLossSpec, accuLossFinal, accuLossPathDiff, accuLossPathSpec = accuLossDiff/(8*len(dataloader)), accuLossSpec/(8*len(dataloader)), accuLossFinal/(8*len(dataloader)), accuLossPathDiff/(8*len(dataloader)), accuLossPathSpec/(8*len(dataloader))
            writer.add_scalar('Train total loss', accuLossFinal if accuLossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train diffuse loss', accuLossDiff if accuLossDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train specular loss', accuLossSpec if accuLossSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train path diffuse loss', accuLossPathDiff if accuLossPathDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train path specular loss', accuLossPathSpec if accuLossPathSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)

        if not os.path.exists('trained_model/' + save_dir):
            os.makedirs('trained_model/' + save_dir)
            print('MAKE DIR {}'.format('trained_model/'+save_dir))

        torch.save({
                'epoch': epoch,
                'model_state_dict': diffuseNet.state_dict(),
                'optimizer_state_dict': optimizerDiff.state_dict(),
                }, 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': specularNet.state_dict(),
                'optimizer_state_dict': optimizerSpec.state_dict(),
                }, 'trained_model/'+ save_dir + '/spec_e{}.pt'.format(epoch+1))

        if use_llpm_buf:
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
        models = {'diffuse': diffuseNet, 'specular': specularNet}
        if use_llpm_buf:
            models['path_diffuse'] = diffPathNet
            models['path_specular'] = specPathNet
        validLossDiff, validLossSpec, validLossFinal, relL2LossFinal, pathDiffLoss, pathSpecLoss = validation(models, validDataloader, eps, criterion, device, epoch, use_llpm_buf,mode)
        writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid total loss', validLossFinal if accuLossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid diffuse loss', validLossDiff if accuLossDiff != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid specular loss', validLossSpec if accuLossSpec != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid path diffuse loss', pathDiffLoss if pathDiffLoss != float('inf') else 0, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid path specular loss', pathSpecLoss if pathSpecLoss != float('inf') else 0, (epoch + 1) * len(dataloader))


        print("Epoch {}".format(epoch + 1))
        print("LossDiff: {}".format(accuLossDiff))
        print("LossSpec: {}".format(accuLossSpec))
        print("LossFinal: {}".format(accuLossFinal))
        print("pathDiffLoss: {}".format(pathDiffLoss))
        print("pathSpecLoss: {}".format(pathSpecLoss))
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
        accuLossPathDiff = 0
        accuLossPathSpec = 0

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
        args.do_finetune,
        args.use_llpm_buf,
        args.manif_w
        )
  

if __name__ == '__main__':
  main()