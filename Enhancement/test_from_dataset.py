# DQPGT: Dynamic Quadruple Priors Guided Transformer for Low-light Image Enhancement

from ast import arg
from copy import deepcopy
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
from skimage import metrics

from basicsr.models import create_model
from basicsr.metrics import calculate_psnr, calculate_ssim
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

def process_and_save_weights(weight_tensor, save_path, prior_name):
    """
    处理权重图：切割4x4网格，计算子图均值，保存带标注的图像
    weight_tensor: [1,1,H,W] 或 [H,W]
    save_path: 保存根目录
    prior_name: 先验名称（如 weights_H）
    """
    # 调整张量维度
    if weight_tensor.dim() == 4:
        weight_map = weight_tensor[0,0].detach().cpu().numpy()  # [H,W]
    else:
        weight_map = weight_tensor.detach().cpu().numpy()
    
    # 创建先验专属目录
    prior_dir = os.path.join(save_path, prior_name)
    os.makedirs(prior_dir, exist_ok=True)
    
    # 获取图像尺寸
    h, w = weight_map.shape
    grid_size = 6
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # 创建画布
    fig = plt.figure(figsize=(20,20), dpi=300)
    plt.imshow(weight_map, cmap='gray')
    plt.axis('off')
    
    # 绘制网格线并计算均值
    means = []
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算子图范围
            y_start = i * cell_h
            y_end = (i+1)*cell_h if i != grid_size-1 else h
            x_start = j * cell_w
            x_end = (j+1)*cell_w if j != grid_size-1 else w
            
            # 计算均值
            cell = weight_map[y_start:y_end, x_start:x_end]
            mean_val = np.mean(cell)
            means.append(mean_val)
            
            # 绘制网格线
            plt.plot([x_start, x_start], [0, h], 'w', linewidth=0.5)
            plt.plot([x_end, x_end], [0, h], 'w', linewidth=0.5)
            plt.plot([0, w], [y_start, y_start], 'w', linewidth=0.5)
            plt.plot([0, w], [y_end, y_end], 'w', linewidth=0.5)
            
            # 添加文本标注
            plt.text(
                x=(x_start + x_end)/2,         # x坐标居中
                y=y_end - cell_h*0.25,         # y坐标靠近底部
                s=f'{mean_val:.3f}', 
                color='white', 
                fontsize=30,
                ha='center',                  # 水平居中
                va='top',                     # 垂直顶部对齐
                bbox=dict(
                    facecolor='black', 
                    alpha=0.7, 
                    pad=2,
                    edgecolor='white',         # 文本框白边
                    boxstyle='round,pad=0.2'
                )
             )
    
    # 保存带标注的全图
    plt.savefig(os.path.join(prior_dir, 'annotated_grid.png'), 
               bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 保存各个子图
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20,20))
    for idx, ax in enumerate(axs.flat):
        i = idx // grid_size
        j = idx % grid_size
        
        y_start = i * cell_h
        y_end = (i+1)*cell_h if i != grid_size-1 else h
        x_start = j * cell_w
        x_end = (j+1)*cell_w if j != grid_size-1 else w
        
        cell = weight_map[y_start:y_end, x_start:x_end]
        ax.imshow(cell, cmap='gray')
        ax.set_title(f'Mean: {means[idx]:.3f}', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(prior_dir, 'subplots.png'), 
               bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 保存均值矩阵
    np.savetxt(os.path.join(prior_dir, 'mean_matrix.txt'), 
              np.array(means).reshape(grid_size, grid_size), 
              fmt='%.3f')


def visualize_feature(feature, filename):
    """
    可视化特征图
    feature: Tensor [C,H,W] 或 [B,C,H,W]
    filename: 保存路径
    """
    if feature.dim() == 4:
        feature = feature[0]  # 取第一个样本
    feature = feature.detach().cpu().float()
    
    if feature.shape[0] == 3:  # RGB通道
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        feature = feature.numpy().transpose(1, 2, 0)
        feature = (feature * 255).astype(np.uint8)
        cv2.imwrite(filename, cv2.cvtColor(feature, cv2.COLOR_RGB2BGR))
        return
    if feature.shape[0] == 1:  # 单通道直接保存灰度
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        feature = (feature.squeeze().numpy() * 255).astype(np.uint8)
        cv2.imwrite(filename, feature)
        return
   # 多通道处理：取通道均值
    if feature.shape[0] > 1:
        feature = torch.mean(feature, dim=0, keepdim=True)
    
    
    # 归一化
    feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
    feature = feature.numpy().transpose(1, 2, 0)  # HWC
    feature = (feature * 255).astype(np.uint8)
    
    # 保存
    if feature.shape[2] == 1:
        cv2.imwrite(filename, feature[:, :, 0])
    else:
        cv2.imwrite(filename, cv2.cvtColor(feature, cv2.COLOR_RGB2BGR))
    
    
parser = argparse.ArgumentParser(
    description='Image Enhancement using DQPGT')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/QuadPriorFormer_LOL_v1.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/LOL_v1.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='LOL_v1', type=str,
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
parser.add_argument('--save_features', action='store_true',
                    help='Save intermediate feature maps')
parser.add_argument('--feature_dir', default='features',
                    type=str, help='Directory for saving features')


args = parser.parse_args()

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


def select_eval_dataset_opt(options):
    """Prefer a dedicated test split while remaining compatible with old val YAMLs."""
    datasets = options.get('datasets', {})
    if 'test' in datasets:
        return deepcopy(datasets['test'])
    if 'val' in datasets:
        return deepcopy(datasets['val'])
    if len(datasets) == 1:
        return deepcopy(next(iter(datasets.values())))
    raise KeyError(
        'No evaluation dataset found. Expected datasets.test or datasets.val.')


eval_dataset_opt = select_eval_dataset_opt(opt)


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
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    eval_dataset_opt['phase'] = 'test'
    if eval_dataset_opt.get('scale') is None:
        eval_dataset_opt['scale'] = 1
    if '~' in eval_dataset_opt['dataroot_gt']:
        eval_dataset_opt['dataroot_gt'] = os.path.expanduser('~') + eval_dataset_opt['dataroot_gt'][1:]
    if '~' in eval_dataset_opt['dataroot_lq']:
        eval_dataset_opt['dataroot_lq'] = os.path.expanduser('~') + eval_dataset_opt['dataroot_lq'][1:]
    dataset = Dataset(eval_dataset_opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
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
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(calculate_psnr(
                target, restored, crop_border=0, input_order='HWC',
                test_y_channel=False))
            ssim.append(calculate_ssim(
                target, restored, crop_border=0, input_order='HWC',
                test_y_channel=False))
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:

    input_dir = eval_dataset_opt['dataroot_lq']
    target_dir = eval_dataset_opt['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

    target_paths = natsorted(glob(os.path.join(
        target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

    if len(input_paths) != len(target_paths):
        raise RuntimeError(
            f'LQ/GT image count mismatch: {len(input_paths)} vs {len(target_paths)}')
    mismatched_pairs = [
        (os.path.basename(inp), os.path.basename(tar))
        for inp, tar in zip(input_paths, target_paths)
        if os.path.basename(inp) != os.path.basename(tar)
    ]
    if mismatched_pairs:
        raise RuntimeError(f'LQ/GT filename mismatch, first pair: {mismatched_pairs[0]}')

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 4
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            
            
            
            
            model_output = None
            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    model_restoration.module.set_test_mode(False)
                    restored = self_ensemble(input_, model_restoration)
                else:
                    model_restoration.module.set_test_mode(args.save_features)
                    model_output = model_restoration(input_)
                    restored = model_output['final_output'] if isinstance(model_output, dict) else model_output
            else:
                # split and test; feature export is disabled for tiled inference.
                model_restoration.module.set_test_mode(False)
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

                
                
            # 保存中间特征
            if args.save_features and isinstance(model_output, dict):
                # 创建特征保存目录
                base_name = os.path.splitext(os.path.basename(inp_path))[0]
                save_dir = os.path.join(args.result_dir, 'features', base_name)
                os.makedirs(save_dir, exist_ok=True)
            
                for stage_idx, stage_out in enumerate(model_output['stage_outputs']):
                    stage_dir = os.path.join(save_dir, f'stage_{stage_idx}')
                    os.makedirs(stage_dir, exist_ok=True)

                    # 单独处理四个权重图
                    weight_maps = {
                        'weights_H': stage_out['weights'][:, 0:1],  # [B,1,H,W]
                        'weights_S': stage_out['weights'][:, 1:2],
                        'weights_RGB': stage_out['weights'][:, 2:3],
                        'weights_Ww': stage_out['weights'][:, 3:4]
                    }
                    # 对每个先验进行处理
                    for prior_name, weight_tensor in weight_maps.items():
                        process_and_save_weights(
                            weight_tensor, 
                            os.path.join(stage_dir, 'weight_grids'), 
                            prior_name
                        )
                    features_to_save = {
                        'weights_H': stage_out['weights'][:, 0:1],  # 拆分各通道
                        'weights_S': stage_out['weights'][:, 1:2],
                        'weights_RGB': stage_out['weights'][:, 2:3],
                        'weights_Ww': stage_out['weights'][:, 3:4],
                        'H_fea': stage_out['H_fea'],
                        'S_fea': stage_out['S_fea'],
                        'RGB_fea': stage_out['RGB_fea'],
                        'RGB_order': stage_out['RGB_order'],
                        'Ww_fea': stage_out['Ww_fea'],
                        'H_fea_w': stage_out['H_fea_w'],
                        'S_fea_w': stage_out['S_fea_w'],
                        'RGB_fea_w': stage_out['RGB_fea_w'],
                        'Ww_fea_w': stage_out['Ww_fea_w'],
                        'features': stage_out['features'],
                        'output': stage_out['output']
                    }
                    for name, feature in features_to_save.items():
                        visualize_feature(feature, os.path.join(stage_dir, f"{name}.png"))
                    qp_msa_features = stage_out.get('QP_MSA_features', {})
                    for name, feature in qp_msa_features.items():
                        # 处理不同特征的形状
                        if name in ['q_inp', 'k_inp', 'v_inp']:
                            # 形状 [b, n, c] → 转为 [b, c, h, w]
                            b, n, c = feature.shape
                            h = int(math.sqrt(n))
                            w = h
                            feature = feature.view(b, h, w, c).permute(0, 3, 1, 2)
                            visualize_feature(feature, os.path.join(stage_dir, f"QP_MSA_{name}.png"))
                        elif name == 'QK_attn':
                            # 注意力矩阵：取第一个样本和第一个头，转为热力图
                            feature = feature[0, 0].detach().cpu().numpy()  # [heads, n, n] → [n, n]
                            plt.figure(frameon=False)  # 关闭边框
                            plt.imshow(feature, cmap='viridis')
                            plt.axis('off')  # 关闭坐标轴
                            plt.gca().set_axis_off()  # 双重保障
                            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去除边距
                            plt.margins(0, 0)  # 去除边距
                            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 关闭刻度
                            plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 关闭刻度
                            plt.savefig(os.path.join(stage_dir, f"QP_MSA_attn_heatmap.png"), bbox_inches='tight', pad_inches=0)
                            plt.close()
                        elif name in ['attn_output', 'x_in']:
                            # 输出特征 [b, h, w, c] → [b, c, h, w]
                            feature = feature.permute(0, 3, 1, 2)
                            visualize_feature(feature, os.path.join(stage_dir, f"QP_MSA_{name}.png"))
                # 保存最终输出
                visualize_feature(model_output['final_output'], 
                            os.path.join(save_dir, 'final_output.png'))
            
            
            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(calculate_psnr(
                target, restored, crop_border=0, input_order='HWC',
                test_y_channel=False))
            ssim.append(calculate_ssim(
                target, restored, crop_border=0, input_order='HWC',
                test_y_channel=False))
            if output_dir != '':
                utils.save_img((os.path.join(output_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            else:
                utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
