import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict

from basicsr.models.losses.loss_util import weighted_loss
from basicsr.models.losses.loss_util import reduce_loss
try:
    from torchvision.models import VGG16_Weights, vgg16
except ImportError:  # torchvision < 0.13
    from torchvision.models import vgg16
    VGG16_Weights = None


_reduction_modes = ['none', 'mean', 'sum']

def _ssim_window(window_size, channel, dtype, device):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / 2.0 / 1.5 ** 2)
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] @ gauss[None, :]
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """返回稳定的 SSIM 相似度；外部使用 1 - SSIM 构造损失。"""

    def __init__(self, window_size=11, size_average=True,
                 data_range=1.0, eps=1e-12):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.data_range = float(data_range)
        self.eps = float(eps)

    def forward(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError(
                f'SSIM input shape mismatch: {img1.shape} vs {img2.shape}')

        # SSIM 的局部统计强制使用 FP32，避免 AMP/FP16 方差误差。
        with torch.cuda.amp.autocast(enabled=False):
            # SSIM 的 C1/C2 按 data_range=1 定义，因此输入也必须在 [0, 1]。
            # 原始 L1 分支仍使用未截断的 pred，可以负责把越界输出拉回来。
            x = img1.float().clamp(0.0, self.data_range)
            y = img2.float().clamp(0.0, self.data_range)

            channel = x.size(1)
            window = _ssim_window(
                self.window_size, channel, x.dtype, x.device)
            padding = self.window_size // 2

            mu1 = F.conv2d(
                x, window, padding=padding, groups=channel)
            mu2 = F.conv2d(
                y, window, padding=padding, groups=channel)

            mu1_sq = mu1.square()
            mu2_sq = mu2.square()
            mu1_mu2 = mu1 * mu2

            sigma1_sq = (
                F.conv2d(
                    x.square(), window,
                    padding=padding, groups=channel)
                - mu1_sq
            ).clamp_min(0.0)

            sigma2_sq = (
                F.conv2d(
                    y.square(), window,
                    padding=padding, groups=channel)
                - mu2_sq
            ).clamp_min(0.0)

            sigma12 = (
                F.conv2d(
                    x * y, window,
                    padding=padding, groups=channel)
                - mu1_mu2
            )

            c1 = (0.01 * self.data_range) ** 2
            c2 = (0.03 * self.data_range) ** 2

            numerator = (
                (2.0 * mu1_mu2 + c1)
                * (2.0 * sigma12 + c2)
            )
            denominator = (
                (mu1_sq + mu2_sq + c1)
                * (sigma1_sq + sigma2_sq + c2)
            ).clamp_min(self.eps)

            ssim_map = numerator / denominator

            # 最终保险：保证 SSIM 有限且位于理论范围。
            ssim_map = torch.nan_to_num(
                ssim_map,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            ).clamp(-1.0, 1.0)

            if self.size_average:
                return ssim_map.mean()

            return ssim_map.mean(dim=(1, 2, 3))
        

@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class L1SSIMLoss(nn.Module):
    """组合 L1 和 SSIM 的损失函数"""
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, alpha=0.3, reduction='mean'):
        super(L1SSIMLoss, self).__init__()
        self.l1_loss = L1Loss(loss_weight=l1_weight, reduction=reduction)  # L1 损失
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True)  # SSIM 损失
        self.alpha = alpha  # L1 和 SSIM 的权重平衡参数
        self.ssim_weight = ssim_weight

    def forward(self, pred, target, weight=None, **kwargs):
        # 计算 L1 损失
        l1 = self.l1_loss(pred, target, weight)
        # 计算 SSIM 损失（SSIM 返回值在 [0, 1]，1 表示完全相似，需转换为损失形式）
        ssim = self.ssim_loss(pred, target)
        total_loss = self.alpha * l1 + (1 - self.alpha) * self.ssim_weight * (1 - ssim)
        return total_loss    


    
class PerceptualLoss(nn.Module):
    """感知损失，使用 VGG16 的中间层特征"""
    def __init__(self, loss_weight=1.0, reduction='mean', layer='relu3_3'):
        super(PerceptualLoss, self).__init__()
        if VGG16_Weights is None:
            vgg = vgg16(pretrained=True).features
        else:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        if layer == 'relu3_3':
            self.vgg = vgg[:16].eval()  # relu3_3 对应 VGG16 的第 16 层
        elif layer == 'relu4_3':
            self.vgg = vgg[:23].eval()  # relu4_3 对应 VGG16 的第 23 层
        else:
            raise ValueError(f'Unsupported layer: {layer}')
        
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结 VGG 参数

        self.register_buffer(
            'imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
            pred_safe = pred.float().clamp(0.0, 1.0)
            target_safe = target.float().clamp(0.0, 1.0)

            mean = self.imagenet_mean.float()
            std = self.imagenet_std.float()

            pred_normalized = (pred_safe - mean) / std
            target_normalized = (target_safe - mean) / std

            pred_fea = self.vgg(pred_normalized)
            target_fea = self.vgg(target_normalized)

            loss = F.mse_loss(
                pred_fea, target_fea, reduction='none')
            return self.loss_weight * reduce_loss(
                loss, self.reduction)
    
class L1PerceptualLoss(nn.Module):
    """组合 L1 Loss 和 Perceptual Loss 的损失函数"""
    def __init__(self, l1_weight=1.0, perc_weight=0.1, reduction='mean'):
        super(L1PerceptualLoss, self).__init__()
        self.l1_loss = L1Loss(loss_weight=l1_weight, reduction=reduction)
        self.perceptual_loss = PerceptualLoss(loss_weight=perc_weight, reduction=reduction)

    def forward(self, pred, target, weight=None):
        l1 = self.l1_loss(pred, target, weight)
        perc = self.perceptual_loss(pred, target)
        total_loss = l1 + perc
        return total_loss
    
    
class L1PerceptualSSIMLoss(nn.Module):
    """组合 L1 Loss、Perceptual Loss 和 SSIM Loss 的损失函数"""
    returns_components = True

    def __init__(self, l1_weight=1.0, perc_weight=0.1, ssim_weight=0.3,
                 layer='relu3_3', reduction='mean'):
        super(L1PerceptualSSIMLoss, self).__init__()
        self.l1_loss = L1Loss(loss_weight=l1_weight, reduction=reduction)  # L1 损失
        self.perceptual_loss = (PerceptualLoss(
            loss_weight=perc_weight, reduction=reduction, layer=layer)
            if perc_weight > 0 else None)
        self.ssim_module = (
            SSIMLoss(window_size=11, size_average=True, data_range=1.0)
            if ssim_weight > 0
            else None
        )
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight
        self.ssim_weight = ssim_weight  # SSIM 损失权重

    def forward(self, pred, target, weight=None, return_components=False):
        """
        Args:
            pred (Tensor): 预测图像，形状 (N, 3, H, W)
            target (Tensor): 目标图像，形状 (N, 3, H, W)
            weight (Tensor, optional): 元素级权重，形状 (N, C, H, W)
        Returns:
            Tensor: 总损失（L1 + Perceptual + SSIM）
        """
        l1 = self.l1_loss(pred, target, weight)  # 计算 L1 损失
        perc = (self.perceptual_loss(pred, target) if self.perceptual_loss is not None
                else pred.new_zeros(()))
        
        if self.ssim_module is not None:
            ssim_score = self.ssim_module(pred, target)
            ssim_loss = 1.0 - ssim_score
            weighted_ssim = self.ssim_weight * ssim_loss
        else:
        # 真正不计算 SSIM，避免 0 * inf 或 0 * NaN。
            ssim_loss = pred.new_zeros(())
            weighted_ssim = pred.new_zeros(())

        total_loss = l1 + perc + weighted_ssim
        if not return_components:
            return total_loss
        components = OrderedDict(
            l1_raw=l1 / self.l1_weight if self.l1_weight else l1.detach() * 0,
            l1_weighted=l1,
            ssim_raw=ssim_loss,
            ssim_weighted=weighted_ssim,
            perceptual_raw=(perc / self.perc_weight
                            if self.perc_weight else perc.detach() * 0),
            perceptual_weighted=perc,
            pred_out_of_range=((pred < 0) | (pred > 1)).float().mean())
        return total_loss, components
    
    
    
    
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss
