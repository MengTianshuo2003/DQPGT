from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2
import math
import matplotlib.pyplot as plt  

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils


from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default=None, type=str,
                    help='Directory containing no-reference test images (required for no-reference datasets)')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
# DQPGT unified evaluation protocol: GT-mean is disabled by default.
gt_mean_group = parser.add_mutually_exclusive_group()
gt_mean_group.add_argument('--GT_mean', dest='GT_mean', action='store_true',
                           help='Enable GT-mean output rectification (diagnostic only)')
gt_mean_group.add_argument('--no_GT_mean', dest='GT_mean', action='store_false',
                           help='Disable GT-mean output rectification (default)')
parser.set_defaults(GT_mean=False)
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')

args = parser.parse_args()

paired_datasets = {'SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor'}
input_dir = None
input_paths = None
if args.dataset not in paired_datasets:
    if not args.input_dir:
        parser.error('--input_dir is required for no-reference datasets.')

    # Select no-reference data explicitly at test time. Reading a shared YAML
    # path here would allow different --dataset labels to process the same data.
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    if not os.path.isdir(input_dir):
        parser.error(
            f'input directory for {args.dataset} does not exist: {input_dir}')

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))
    if not input_paths:
        parser.error(
            f'no .png or .jpg test images found for {args.dataset} in: {input_dir}')

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
output_dir = args.output_dir

os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

if dataset in paired_datasets:
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    if '~' in opt['dataroot_gt']:
        opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
else:

    print(f'Input directory: {input_dir}')
    print(f'Found {len(input_paths)} test images for {args.dataset}.')

    with torch.inference_mode():
        for inp_path in tqdm(input_paths):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 4
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    model_restoration.module.set_test_mode(False)
                    restored = self_ensemble(input_, model_restoration)
                else:
                    model_restoration.module.set_test_mode(True)
                    model_output = model_restoration(input_)
                    restored = model_output['final_output']
            else:
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if output_dir != '':
                utils.save_img((os.path.join(output_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            else:
                utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
