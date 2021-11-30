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

save_dir = 'kpcn_decomp_single'
writer = SummaryWriter('kpcn/'+save_dir)
def _gradients(buf):
        """Compute the xy derivatives of the input buffer. This helper is used in the _preprocess_<base_model>(...) functions
        Args:
            buf(np.array)[B, C, H, W]: input image-like tensor.
        Returns:
            (np.array)[B, C, H, W]: horizontal and vertical gradients of buf.
        """
        dx = buf[:, :, :, 1:] - buf[:, :, :, :-1]
        dy = buf[:, :, 1:] - buf[:, :, :-1]
        dx = F.pad(dx, (1,0), "constant", 0.0) # zero padding to the leftni
        dy = F.pad(dy, (0,0,1,0), 'constant', 0.0)  # zero padding to the up
        # print(dx.shape, dy.shape)
        return torch.cat([dx, dy], 1)

# make diffuse&specular batch into one single image batch
def make_batch(batch):
    pass
    noisy = batch['kpcn_diffuse_buffer'] * (batch['kpcn_albedo'] + eps) + torch.exp(batch['kpcn_specular_buffer']) - 1.0
    noisy_grad = _gradients(noisy)
    gbuffers = batch['kpcn_diffuse_in'][:, 10:-1]
    # print(noisy.shape, noisy_grad.shape, gbuffers.shape)
    inp = torch.cat((noisy, noisy_grad, gbuffers), dim=1)
    batch['kpcn_in'] = inp
    return batch, noisy


def validation(models, dataloader, eps, criterion, device, epoch, use_llpm_buf, mode='kpcn'):
    pass
    lossGbuffer = 0
    lossPbuffer = 0
    lossFinal = 0
    relL2Final = 0
    lossPath = 0
    relL2 = RelativeMSE()
    path_criterion = GlobalRelativeSimilarityLoss()
    # for batch_idx, data in enumerate(dataloader):
    batch_idx = 0
    decompNet, gbufferNet, pbufferNet, pathNet = models['decomp'], models['gbuffer'], models['pbuffer'], models['path']
    decompNet.eval(), gbufferNet.eval(), pbufferNet.eval(), pathNet.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, ncols=70):
             
            # Decompose image
            # print(batch['kpcn_specular_in'].shape)
            for k, v in batch.items():
                batch[k] = v.to(device)
            batch, noisy = make_batch(batch)
            mask, f1, f2 = decompNet(batch['kpcn_in'])
            target = batch['target_total']

            # if use_llpm_buf:
            paths = batch['paths'].to(device)
            p_buffer = pathNet(paths)
            '''Feature Disentanglement'''    
            #TODO
            _, _, c, _, _ = p_buffer.shape
            assert c >= 2
            
            # Variance
            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= p_buffer.shape[1]
            # print(p_buffer.shape, p_buffer.mean(1).shape, p_var.shape)
            pbuffer = torch.cat((p_buffer.mean(1), p_var), dim=1)
            f2 = torch.cat((batch['kpcn_diffuse_in'][:,-1].unsqueeze(1), f2, pbuffer), dim=1)
            # Denosing using only G-buffers

            # g_buffer input
            g_input = noisy * (torch.ones_like(mask) - mask)
            g_target = target * (torch.ones_like(mask) - mask)
            g_output = gbufferNet(torch.cat((torch.log(g_input + 1.0), f1), dim=1), g_input)
            g_target = crop_like(g_target, g_output)
            lossGbuffer += criterion(g_target, g_output).item()

            # p_buffer input
            p_input = noisy * mask
            p_target = target * mask
            p_output = pbufferNet(torch.cat((torch.log(p_input + 1.0), f2), dim=1), p_input)
            p_target = crop_like(p_target, p_output)
            lossPbuffer += criterion(p_target, p_output).item()

            # Loss of merged denoised result
            outputFinal = g_output + p_output
            target = crop_like(target, outputFinal)
            lossFinal += criterion(outputFinal, target).item()
            relL2Final += relL2(outputFinal, target).item()

            # if use_llpm_buf:
            p_buffer = crop_like(p_buffer, outputFinal)
            loss_manif = path_criterion(p_buffer, p_target)
            lossPath += loss_manif
                # lossDiff += 0.1 * loss_manif_diffuse
                # lossSpec += 0.1 * loss_manif_specular

            # visualize
            if batch_idx == 20:
                # inputFinal = batch['kpcn_diffuse_buffer'] * (batch['kpcn_albedo'] + eps) + torch.exp(batch['kpcn_specular_buffer']) - 1.0
                inputGrid = torchvision.utils.make_grid(noisy)
                writer.add_image('noisy patches e{}'.format(epoch+1), inputGrid)
                writer.add_image('noisy patches e{}'.format(str(epoch+1)+'_'+str(batch_idx)), inputGrid)

                outputGrid = torchvision.utils.make_grid(outputFinal)
                writer.add_image('denoised patches e{}'.format(str(epoch+1)+'_'+str(batch_idx)), outputGrid)
                # writer.add_image('denoised patches e{}'.format(epoch+1), outputGrid)

                cleanGrid = torchvision.utils.make_grid(target)
                # writer.add_image('clean patches e{}'.format(epoch+1), cleanGrid)
                writer.add_image('clean patches e{}'.format(str(epoch+1)+'_'+str(batch_idx)), cleanGrid)

            batch_idx += 1

    return lossGbuffer/(4*len(dataloader)), lossPbuffer/(4*len(dataloader)), lossFinal/(4*len(dataloader)), relL2Final/(4*len(dataloader)), lossPath/(4*len(dataloader))


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
    decompNet = decompOriginModule(in_channel=33, discrete=do_discrete).to(device)
    optimizerDecomp = optim.Adam(decompNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    gbufferNet = KPCN(L, 67, hidden_channels, kernel_size).to(device)
    pbufferNet = KPCN(L, 72, hidden_channels, kernel_size).to(device)

    print('LEARNING RATE : {}'.format(learning_rate))
    optimizerGbuffer = optim.Adam(gbufferNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    optimizerPbuffer = optim.Adam(pbufferNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    pathNet = PathNet(trainset.pnet_in_size).to(device)
    optimizerPath = optim.Adam(pathNet.parameters(), lr=1e-4, betas=(0.9, 0.99))
    path_criterion = GlobalRelativeSimilarityLoss()


    # print(decompNet, "CUDA:", next(decompNet.parameters()).device)
    # print(gbufferNet, "CUDA:", next(gbufferNet.parameters()).device)
    # print(pbufferNet, "CUDA:", next(pbufferNet.parameters()).device)
    print('# Parameter for DecompNet : {}'.format(sum([p.numel() for p in decompNet.parameters()])))
    print('# Parameter for GbufferNet : {}'.format(sum([p.numel() for p in gbufferNet.parameters()])))
    print('# Parameter for PbufferNet : {}'.format(sum([p.numel() for p in pbufferNet.parameters()])))
    print('# Parameter for PathNet : {}'.format(sum([p.numel() for p in pathNet.parameters()])))
    # print(summary(diffuseNet, input_size=(3, 128, 128)))

    if loss == 'L1':
        criterion = nn.L1Loss()
    elif loss =='SMAPE':
        criterion = SMAPE()
    else:
        print('Loss Not Supported')
        return
    # checkpointDiffPath = torch.load('trained_model/kpcn_decomp_3/path_diff_e5.pt')
    # diffPathNet.load_state_dict(checkpointDiffPath['model_state_dict'])
    # optimizerDiffPath.load_state_dict(checkpointDiffPath['optimizer_state_dict'])
    pathNet.train()

    # checkpointDiff1 = torch.load('trained_model/kpcn_decomp_3/diff1_e5.pt')
    # diffuseNet1.load_state_dict(checkpointDiff1['model_state_dict'])
    # optimizerDiff1.load_state_dict(checkpointDiff1['optimizer_state_dict'])
    gbufferNet.train()

    # checkpointSpec1 = torch.load('trained_model/kpcn_decomp_3/spec1_e5.pt')
    # specularNet1.load_state_dict(checkpointSpec1['model_state_dict'])
    # optimizerSpec1.load_state_dict(checkpointSpec1['optimizer_state_dict'])
    pbufferNet.train()

    # pNet.train()

    accuLossGbuffer = 0
    accuLossPbuffer = 0
    accuLossPath = 0
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
        models = {'decomp': decompNet, 
                'gbuffer': gbufferNet,
                'pbuffer': pbufferNet,
                'path': pathNet
                }
        initLossGbuffer, initLossPbuffer, initLossFinal, relL2LossFinal, pathLoss = validation(models, validDataloader, eps, criterion, device, -1, use_llpm_buf,mode)
        print("initLossGbuffer: {}".format(initLossGbuffer))
        print("initLossPbuffer: {}".format(initLossPbuffer))
        print("initLossFinal: {}".format(initLossFinal))
        print("relL2LossFinal: {}".format(relL2LossFinal))
        print("pathLoss: {}".format(pathLoss))
        writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 0, (init_epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid total loss', initLossFinal if initLossFinal != float('inf') else 0, (init_epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid Gbuffer Loss', initLossGbuffer if initLossGbuffer != float('inf') else 0, (init_epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid Pbuffer Loss', initLossPbuffer if initLossPbuffer != float('inf') else 0, (init_epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid Path Loss', pathLoss if pathLoss != float('inf') else 0, (init_epoch + 1) * len(validDataloader))


    import time

    start = time.time()
    print('START')

    for epoch in range(init_epoch, epochs):
        print('EPOCH {}'.format(epoch+1))
        decompNet.train()
        gbufferNet.train()
        pbufferNet.train()
        pathNet.train()
        i_batch = -1
        for batch in tqdm(dataloader, leave=False, ncols=70):
            i_batch += 1
            
            optimizerDecomp.zero_grad()
            optimizerGbuffer.zero_grad()
            optimizerPbuffer.zero_grad()
            optimizerPath.zero_grad()
            
            for k, v in batch.items():
                batch[k] = v.to(device)
            batch, noisy = make_batch(batch)
            mask, f1, f2 = decompNet(batch['kpcn_in'])
            target = batch['target_total']

            # if use_llpm_buf:
            paths = batch['paths'].to(device)
            p_buffer = pathNet(paths)
            '''Feature Disentanglement'''    
            #TODO
            _, _, c, _, _ = p_buffer.shape
            assert c >= 2
            
            # Variance
            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= p_buffer.shape[1]
            # print(p_buffer.shape, p_buffer.mean(1).shape, p_var.shape)
            pbuffer = torch.cat((p_buffer.mean(1), p_var), dim=1)
            f2 = torch.cat((batch['kpcn_diffuse_in'][:,-1].unsqueeze(1), f2, pbuffer), dim=1)
            # Denosing using only G-buffers

            # g_buffer input
            g_input = noisy * (torch.ones_like(mask) - mask)
            g_target = target * (torch.ones_like(mask) - mask)
            g_output = gbufferNet(torch.cat((torch.log(g_input + 1.0), f1), dim=1), g_input)
            g_target = crop_like(g_target, g_output)
            lossGbuffer = criterion(g_target, g_output)

            # p_buffer input
            p_input = noisy * mask
            p_target = target * mask
            p_output = pbufferNet(torch.cat((torch.log(p_input + 1.0), f2), dim=1), p_input)
            p_target = crop_like(p_target, p_output)
            lossPbuffer = criterion(p_target, p_output)

            # Loss of merged denoised result
            # outputFinal = g_output + p_output
            # target = crop_like(target, outputFinal)
            # lossFinal = criterion(outputFinal, target)

            # if use_llpm_buf:
            p_buffer = crop_like(p_buffer, p_target)
            loss_manif = path_criterion(p_buffer, p_target)
            lossPbuffer += 0.1 * loss_manif
                

            # if not do_finetune:
            #     lossGbuffer.backward(retain_graph=True)
            #     optimizerGbuffer.step()
            #     lossPbuffer.backward()
            #     optimizerPbuffer.step()
            #     optimizerPath.step()
            #     optimizerDecomp.step()

            #     # Loss of merged denoised result
            #     with torch.no_grad():
            #         outputFinal = g_output + p_output
            #         target = crop_like(target, outputFinal)
            #         lossFinal = criterion(outputFinal, target).item()

            # if do_finetune:
            #     # print('FINETUNING')
            #     outputFinal = g_output + p_output
            #     target = crop_like(target, outputFinal)
            #     lossFinal = criterion(outputFinal, target).item()
            #     lossFinal += 0.1 * loss_manif
            #     lossFinal.backward()
            #     optimizerDecomp.step()
            #     optimizerGbuffer.step()
            #     optimizerPbuffer.step()
            #     optimizerPath.step()
            
            outputFinal = g_output + p_output
            target = crop_like(target, outputFinal)
            lossFinal = criterion(outputFinal, target).item()
            lossFinal += 0.1 * loss_manif
            lossFinal.backward()
            optimizerDecomp.step()
            optimizerGbuffer.step()
            optimizerPbuffer.step()
            optimizerPath.step()
            
            accuLossGbuffer += lossGbuffer
            accuLossPbuffer += lossPbuffer
            accuLossPath += loss_manif
            accuLossFinal += lossFinal

            writer.add_scalar('lossFinal', lossFinal if lossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossGbuffer', lossGbuffer if lossGbuffer != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossPbuffer', lossPbuffer if lossPbuffer != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('lossPath', loss_manif if loss_manif != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    
            accuLossGbuffer, accuLossPbuffer, accuLossPath, accuLossSpec2, accuLossFinal = accuLossGbuffer/(8*len(dataloader)), accuLossPbuffer/(8*len(dataloader)), accuLossPath/(8*len(dataloader)), accuLossSpec2/(8*len(dataloader)), accuLossFinal/(8*len(dataloader))
            writer.add_scalar('Train total loss', accuLossFinal if accuLossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train gbuffer loss', accuLossGbuffer if accuLossGbuffer != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train pbuffer loss', accuLossPbuffer if accuLossPbuffer != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
            writer.add_scalar('Train path loss', accuLossPath if accuLossPath != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)


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
                'model_state_dict': gbufferNet.state_dict(),
                'optimizer_state_dict': optimizerGbuffer.state_dict(),
                }, 'trained_model/'+ save_dir + '/gbuffer_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': pbufferNet.state_dict(),
                'optimizer_state_dict': optimizerPbuffer.state_dict(),
                }, 'trained_model/'+ save_dir + '/pbuffer_e{}.pt'.format(epoch+1))
        torch.save({
                'epoch': epoch,
                'model_state_dict': pathNet.state_dict(),
                'optimizer_state_dict': optimizerPath.state_dict(),
                }, 'trained_model/'+ save_dir + '/path_e{}.pt'.format(epoch+1))
        # print('VALIDATION WORKING!')
        models = {'decomp': decompNet, 
                'gbuffer': gbufferNet,
                'pbuffer': pbufferNet,
                'path': pathNet
                }
        validLossGbuffer, validLossPbuffer, validLossFinal, relL2LossFinal, pathLoss = validation(models, validDataloader, eps, criterion, device, epoch, use_llpm_buf,mode)
        writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid total loss', validLossFinal if accuLossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
        writer.add_scalar('Valid gbuffer loss', validLossGbuffer if validLossGbuffer != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid pbuffer loss', validLossPbuffer if validLossPbuffer != float('inf') else 0, (epoch + 1) * len(validDataloader))
        writer.add_scalar('Valid path loss', pathLoss if pathLoss != float('inf') else 0, (epoch + 1) * len(dataloader))


        print("Epoch {}".format(epoch + 1))
        print("ValidLossGbuffer: {}".format(validLossGbuffer))
        print("ValidLossPbuffer: {}".format(validLossPbuffer))
        print("ValidLossPath: {}".format(pathLoss))
        print("ValidLossFinal: {}".format(validLossFinal))
        print("ValidrelL2LossDiff: {}".format(relL2LossFinal))
        

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

        print('SAVED {} epoch e{}'.format(save_dir, epoch+1))

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