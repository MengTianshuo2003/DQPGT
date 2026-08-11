import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.training_diagnostics import TrainingDiagnostics

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')
                  
        # Mixup is a training-only augmentation. Test configurations are not
        # required to contain train.mixing_augs.
        mixing_opt = self.opt.get('train', {}).get('mixing_augs', {})
        self.mixing_flag = self.is_train and mixing_opt.get('mixup', False)
        if self.mixing_flag:
            mixup_beta = mixing_opt.get('mixup_beta', 1.2)
            use_identity = mixing_opt.get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
            self.diagnostics = TrainingDiagnostics(self.net_g, self.opt)
        else:
            self.diagnostics = None

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)      #如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if self.diagnostics is not None:
            self.diagnostics.begin_iteration(current_iter)

        with autocast(enabled=self.use_amp):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            loss_dict = OrderedDict()
            # pixel loss
            l_pix = 0.
            component_sums = OrderedDict()
            for pred in preds:
                if getattr(self.cri_pix, 'returns_components', False):
                    pred_loss, components = self.cri_pix(
                        pred, self.gt, return_components=True)
                    l_pix += pred_loss
                    for name, value in components.items():
                        component_sums[name] = component_sums.get(name, 0.) + value
                else:
                    l_pix += self.cri_pix(pred, self.gt)

            loss_dict['l_pix'] = l_pix
            for name, value in component_sums.items():
                loss_dict[f'l_{name}'] = value

        if not torch.isfinite(l_pix):
            logger = get_root_logger()
            logger.warning(f'Skip iter {current_iter}: non-finite loss detected, l_pix={l_pix.item()}')
            if self.diagnostics is not None:
                prediction_nonfinite = []
                for index, pred in enumerate(preds):
                    pred_detached = pred.detach()
                    if not torch.isfinite(pred_detached).all().item():
                        prediction_nonfinite.append({
                            'prediction_index': index,
                            'nan_count': torch.isnan(pred_detached).sum().item(),
                            'posinf_count': torch.isposinf(pred_detached).sum().item(),
                            'neginf_count': torch.isneginf(pred_detached).sum().item()
                        })
                self.diagnostics.report_nonfinite(
                    'nonfinite_loss', {
                        'loss': l_pix.detach().float().item(),
                        'loss_module': self.cri_pix.__class__.__name__,
                        'network_output_finite': len(prediction_nonfinite) == 0,
                        'prediction_nonfinite': prediction_nonfinite
                    })
            self.optimizer_g.zero_grad(set_to_none=True)
            loss_dict['l_pix'] = torch.nan_to_num(l_pix.detach(), nan=0.0, posinf=1e4, neginf=-1e4)
            self.log_dict = self.reduce_loss_dict(loss_dict)
            return

        self.amp_scaler.scale(l_pix).backward()
        self.amp_scaler.unscale_(self.optimizer_g) # 在梯度裁剪前先unscale梯度
        # l_pix.backward()

        if self.diagnostics is not None:
            self.diagnostics.collect_gradient_norms(self.net_g)

        if self.opt['train']['use_grad_clip']:
            try:
                # error_if_nonfinite preserves the original gradients, allowing
                # diagnostics to identify the first affected parameters.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net_g.parameters(), 0.01, error_if_nonfinite=True)
            except RuntimeError as error:
                logger = get_root_logger()
                affected = []
                if self.diagnostics is not None:
                    affected = self.diagnostics.find_nonfinite_gradients(self.net_g)
                    self.diagnostics.report_nonfinite(
                        'nonfinite_gradient', affected)
                logger.warning(
                    f'Skip iter {current_iter}: non-finite grad norm detected; '
                    f'affected_params={len(affected)}; error={error}')
                self.optimizer_g.zero_grad(set_to_none=True)
                self.log_dict = self.reduce_loss_dict(loss_dict)
                return
            if self.diagnostics is not None:
                self.diagnostics.set_global_gradient_norm(grad_norm)
        # self.optimizer_g.step()
        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        if self.diagnostics is not None:
            self.diagnostics.finish_iteration()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return {}

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            if idx == 0 and self.diagnostics is not None:
                self.diagnostics.record_validation_probe(
                    self.net_g, current_iter, img_name)

            if (idx == 0 and self.opt.get('is_train') and
                    self.opt['val'].get('loss_probe_gradients', False) and
                    getattr(self.cri_pix, 'returns_components', False)):
                # The first manifest entry is a fixed validation probe. Compute
                # component gradients w.r.t. the unclamped prediction only;
                # this does not backpropagate through or update the network.
                with torch.enable_grad():
                    probe_pred = self.output.detach().clone().requires_grad_(True)
                    _, probe_components = self.cri_pix(
                        probe_pred, self.gt.detach(), return_components=True)
                    probe_log = {}
                    for component_name in ('l1_weighted', 'ssim_weighted',
                                           'perceptual_weighted'):
                        component = probe_components[component_name]
                        if component.requires_grad:
                            gradient = torch.autograd.grad(
                                component, probe_pred, retain_graph=True,
                                allow_unused=True)[0]
                            grad_norm = (0.0 if gradient is None else
                                         gradient.detach().float().norm().item())
                        else:
                            grad_norm = 0.0
                        probe_log[f'{component_name}_grad_norm'] = grad_norm
                    probe_log['pred_out_of_range'] = probe_components[
                        'pred_out_of_range'].detach().float().item()
                logger = get_root_logger()
                logger.info(f'Validation loss probe {img_name}: {probe_log}')
                if tb_logger:
                    for name, value in probe_log.items():
                        tb_logger.add_scalar(f'loss_probe/{name}', value, current_iter)

            visuals = self.get_current_visuals()
            # Unified evaluator contract: float RGB in [0, 1]. The same path
            # is used by validation and basicsr/test.py.
            visuals['result'] = visuals['result'].clamp(0, 1)
            if 'gt' in visuals:
                visuals['gt'] = visuals['gt'].clamp(0, 1)
                if self.opt['val'].get('gt_mean', False):
                    # GT-mean rectification in RGB tensor space. This is
                    # applied identically in validation and basicsr/test.py.
                    rgb_weights = visuals['result'].new_tensor(
                        [0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                    restored_mean = (visuals['result'] * rgb_weights).sum(
                        dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
                    target_mean = (visuals['gt'] * rgb_weights).sum(
                        dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
                    visuals['result'] = (visuals['result'] * (
                        target_mean / restored_mean.clamp_min(1e-8))).clamp(0, 1)
            # OpenCV writer expects BGR uint8; this conversion is for saving
            # only and is never used by the float-RGB metric path below.
            sr_img = tensor2img([visuals['result']], rgb2bgr=True)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=True)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        # Return every metric explicitly. Model selection must name its primary
        # metric instead of depending on YAML insertion order.
        return dict(self.metric_results) if with_metrics else {}

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
