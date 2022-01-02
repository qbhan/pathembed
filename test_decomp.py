from typing import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import argparse
import os
from tqdm import tqdm
import csv
from decomp import decompModule

from utils import *
from kpcn import *
from path import *
from dataset import DenoiseDataset
from losses import *

permutation = [0, 3, 1, 2]
eps = 0.00316

parser = argparse.ArgumentParser(description='Test the model')


parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=34, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)
parser.add_argument('--mask_channel', default=3, type=int)
parser.add_argument('--use_llpm_buf', default=False, type=bool)
parser.add_argument('--decomp')
parser.add_argument('--diffuse1')
parser.add_argument('--specular1')
parser.add_argument('--diffuse2')
parser.add_argument('--specular2')
parser.add_argument('--diffusePath')
parser.add_argument('--specularPath')
parser.add_argument('--path_diffuse_model', default=None)
parser.add_argument('--path_specular_model', default=None)
parser.add_argument('--data_dir')
parser.add_argument('--save_dir')

parser.set_defaults(do_vis_feat=False)
parser.add_argument('--do_vis_feat', dest='do_vis_feat', action='store_true')
# parser.set_defaults(do_errormap=False)
# parser.add_argument('--do_errormap', dest='do_errormap', action='store_true')

save_dir = 'kpcn_decomp_mask'
writer = SummaryWriter('test_runs/'+save_dir)


def denoise(models, dataloader, device, save_dir, do_vis_feat, use_llpm_buf, debug=False):
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
        g_input = torch.zeros((3, 960, 960)).to(device)
        p_input = torch.zeros((3, 960, 960)).to(device)
        g_image = torch.zeros((3, 960, 960)).to(device)
        p_image = torch.zeros((3, 960, 960)).to(device)
        g_denoise = torch.zeros((3, 960, 960)).to(device)
        p_denoise = torch.zeros((3, 960, 960)).to(device)
        mask_image = torch.zeros((3, 960, 960)).to(device)

        decompNet, diffuseNet1, specularNet1, diffuseNet2, specularNet2, diffPathNet, specPathNet = models['decomp'], models['diffuse1'], models['specular1'], models['diffuse2'], models['specular2'], models['path_diffuse'], models['path_specular']

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
        mask_supervision = torch.zeros((3, 960, 960)).to(device)
        x, y = 0, 0
        with open(save_dir + '/error.csv', 'w', encoding='UTF_8') as f:
            csvwriter = csv.writer(f)
            for batch in tqdm(dataloader, leave=False, ncols=70):
                # print(x, y)
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

                outputSpec1 = specularNet1(X_spec1)
                Y_spec1 = crop_like(Y_spec1, outputSpec1)

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

                outputSpec2 = specularNet2(X_spec2)
                Y_spec2 = crop_like(Y_spec2, outputSpec2)

                # calculate final ground truth error
                albedo = batch2['kpcn_albedo'].to(device)
                albedo = crop_like(albedo, outputDiff2)
                outputFinal2 = outputDiff2 * (albedo + eps) + torch.exp(outputSpec2) - 1.0


                # Loss of merged denoised result
                outputFinal = outputFinal1 + outputFinal2



                Y_final = batch['target_total'].to(device)
                Y_final = crop_like(Y_final, outputFinal)
                # print(lossFinal, relL2Final)
                lossFinal += criterion(outputFinal, Y_final).item()
                scenelossFinal += criterion(outputFinal, Y_final).item()
                relL2Final += relL2(outputFinal, Y_final).item()
                scenerelL2Final += relL2(outputFinal, Y_final).item()


                # visualize
                inputFinal = batch['kpcn_diffuse_buffer'] * (batch['kpcn_albedo'] + eps) + torch.exp(batch['kpcn_specular_buffer']) - 1.0
                input1 = inputFinal * mask
                input2 = inputFinal * (1 - mask)

                # print(np.shape(inputFinal))
                # print(np.shape(outputFinal))
                # print(np.shape(Y_final))
                mask_image[:, x*64:x*64+64, y*64:y*64+64] = mask[0, :, 32:96, 32:96]
                g_input[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(input1[0, :, 32:96, 32:96])
                p_input[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(input2[0, :, 32:96, 32:96])
                g_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(batch1['target_total'][0, :, 32:96, 32:96])
                p_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(batch2['target_total'][0, :, 32:96, 32:96])
                g_denoise[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(outputFinal1[0, :, 16:80, 16:80])
                p_denoise[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(outputFinal2[0, :, 16:80, 16:80])
                input_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(inputFinal[0, :, 32:96, 32:96])
                output_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(outputFinal[0, :, 16:80, 16:80])
                gt_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(Y_final[0, :, 16:80, 16:80])
                error_map[:, x*64:x*64+64, y*64:y*64+64] = torch.abs(gt_image[:, x*64:x*64+64, y*64:y*64+64] - output_image[:, x*64:x*64+64, y*64:y*64+64])
                em = torch.abs(gt_image[:, x*64:x*64+64, y*64:y*64+64] - output_image[:, x*64:x*64+64, y*64:y*64+64])
                em = torch.ones_like(em) - F.normalize(em)
                # print(em.shape)
                mask_supervision[:, x*64:x*64+64, y*64:y*64+64] = em


                if do_vis_feat:
                    diff_rad[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,:3,:,:][0, :3, 32:96, 32:96]
                    diff_rad_var[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,3,:,:][0, 32:96, 32:96]
                    diff_rad_dx[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,4:7,:,:][0, :, 32:96, 32:96]
                    diff_rad_dy[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,7:10,:,:][0, :, 32:96, 32:96]
                    spec_rad[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_specular_in'][:,:3,:,:][0, :3, 32:96, 32:96]
                    spec_rad_var[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_specular_in'][:,3,:,:][0, 32:96, 32:96]
                    spec_rad_dx[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_specular_in'][:,4:7,:,:][0, :, 32:96, 32:96]
                    spec_rad_dy[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_specular_in'][:,7:10,:,:][0, :, 32:96, 32:96]
                    normal[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,10:13,:,:][0, :, 32:96, 32:96]
                    normal_var[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,13,:,:][0, 32:96, 32:96]
                    normal_dx[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,14:17,:,:][0, :, 32:96, 32:96]
                    normal_dy[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,17:20,:,:][0, :, 32:96, 32:96]
                    depth[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,20,:,:][0, 32:96, 32:96]
                    depth_var[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,21,:,:][0, 32:96, 32:96]
                    depth_dx[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,22,:,:][0, 32:96, 32:96]
                    depth_dy[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,23,:,:][0, 32:96, 32:96]
                    albedo_in[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,24:27,:,:][0, :, 32:96, 32:96]
                    albedo_in_var[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,27,:,:][0, 32:96, 32:96]
                    albedo_in_dx[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,28:31,:,:][0, :, 32:96, 32:96]
                    albedo_in_dy[:, x*64:x*64+64, y*64:y*64+64] = batch['kpcn_diffuse_in'][:,31:34,:,:][0, :, 32:96, 32:96]

                
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
                    if mask_image.shape[0] == 3:
                        save_image(mask_image[0, :, :], save_dir + '/test{}/mask_r.png'.format(image_idx))
                        save_image(mask_image[1, :, :], save_dir + '/test{}/mask_g.png'.format(image_idx))
                        save_image(mask_image[2, :, :], save_dir + '/test{}/mask_b.png'.format(image_idx))
                    save_image(mask_image, save_dir + '/test{}/mask.png'.format(image_idx))
                    save_image(input_image, save_dir + '/test{}/noisy.png'.format(image_idx))
                    save_image(g_input, save_dir + '/test{}/g_noisy.png'.format(image_idx))
                    save_image(p_input, save_dir + '/test{}/p_noisy.png'.format(image_idx))
                    save_image(output_image, save_dir + '/test{}/denoise.png'.format(image_idx))
                    save_image(gt_image, save_dir + '/test{}/clean.png'.format(image_idx))
                    save_image(g_image, save_dir + '/test{}/g_image.png'.format(image_idx))
                    save_image(p_image, save_dir + '/test{}/p_image.png'.format(image_idx))
                    save_image(g_denoise, save_dir + '/test{}/g_denoise.png'.format(image_idx))
                    save_image(p_denoise, save_dir + '/test{}/p_denoise.png'.format(image_idx))
                    save_image(error_map, save_dir + '/test{}/error_map.png'.format(image_idx))
                    save_image(mask_supervision, save_dir + '/test{}/mask_supervision.png'.format(image_idx))

                    # losses
                    scenelossDiff, scenelossSpec, scenelossFinal, scenerelL2Final = scenelossDiff/225, scenelossSpec/225, scenelossFinal/225, scenerelL2Final/225
                    # print('test {} lossDiff : {}'.format(image_idx, scenelossDiff))
                    # print('test {} lossSpec : {}'.format(image_idx, scenelossSpec))
                    print('test {} lossFinal : {}'.format(image_idx, scenelossFinal))
                    print('test {} relL2Final : {}'.format(image_idx, scenerelL2Final))
                    # writer.add_scalar('test lossDiff',  scenelossDiff if lossFinal != float('inf') else 1e+35, image_idx)
                    # writer.add_scalar('test lossSpec',  scenelossSpec if lossFinal != float('inf') else 1e+35, image_idx)
                    # writer.add_scalar('test lossFinal',  scenelossFinal if lossFinal != float('inf') else 1e+35, image_idx)
                    # writer.add_scalar('test relL2Final',  scenerelL2Final if lossFinal != float('inf') else 1e+35, image_idx)
                    print('write csv')
                    csvwriter.writerow([scenelossFinal, scenerelL2Final])

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


def main():
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DenoiseDataset(args.data_dir, 8, 'kpcn', 'test', 1, 'recon',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
  
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=False
        )
    #   models = {'dif'}

        # diffuseNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
        # specularNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    models = OrderedDict()
    print(25 + args.mask_channel)
    models['decomp'] = decompModule(in_channel=25+args.mask_channel, out_channel=args.mask_channel).to(args.device)
    models['diffuse1'] = KPCN(args.num_layers, 34, args.hidden_channels, args.kernel_size).to(args.device)
    models['specular1']= KPCN(args.num_layers, 34, args.hidden_channels, args.kernel_size).to(args.device)
    models['diffuse2'] = KPCN(args.num_layers, 39, args.hidden_channels, args.kernel_size).to(args.device)
    models['specular2']= KPCN(args.num_layers, 39, args.hidden_channels, args.kernel_size).to(args.device)
    models['path_diffuse'] = PathNet(dataset.pnet_in_size).to(args.device)
    models['path_specular'] = PathNet(dataset.pnet_in_size).to(args.device)

    # print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
    # print(torch.load(args.diffuse_model).keys)
    ckptDecomp = torch.load(args.decomp)
    ckptDiff1 = torch.load(args.diffuse1)
    ckptSpec1 = torch.load(args.specular1)
    ckptDiff2 = torch.load(args.diffuse2)
    ckptSpec2 = torch.load(args.specular2)
    ckptDiffPath = torch.load(args.diffusePath)
    ckptSpecPath = torch.load(args.specularPath)
    models['decomp'].load_state_dict(ckptDecomp['model_state_dict'])
    models['diffuse1'].load_state_dict(ckptDiff1['model_state_dict'])
    models['specular1'].load_state_dict(ckptSpec1['model_state_dict'])
    models['diffuse2'].load_state_dict(ckptDiff2['model_state_dict'])
    models['specular2'].load_state_dict(ckptSpec2['model_state_dict'])
    models['path_diffuse'].load_state_dict(ckptDiffPath['model_state_dict'])
    models['path_specular'].load_state_dict(ckptSpecPath['model_state_dict'])
    
    
    _, _, totalL1, totalrelL2 = denoise(models, dataloader, args.device, args.save_dir, args.do_vis_feat, args.use_llpm_buf)
    print('TEST L1 LOSS is {}'.format(totalL1))
    print('TEST L2 LOSS is {}'.format(totalrelL2))
    # with open(args.save_dir + '/error.csv', 'w', encoding='UTF_8') as f:
    #         csvwriter = csv.writer(f)
    #         csvwriter.writerow([totalL1, totalrelL2])



if __name__ == '__main__':
  main()