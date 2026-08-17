import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2

eps = 1e-4#防止分母为零

# ==================================
# ======== Gaussian filter =========
# ==================================
#该函数用于生成高斯滤波器的基础滤波器。
def gaussian_basis_filters(scale, gpu, k=3):
    std = torch.pow(2,scale) #计算标准差：std = 2^scale（指数关系）

    # Define the basis vector for the current scale
    filtersize = torch.ceil(k*std+0.5) #计算滤波器尺寸：k*std向上取整，保证奇数尺寸

    if torch.isnan(filtersize).any():
        raise ValueError("filtersize为nan值")

    x = torch.arange(start=-filtersize.item(), end=filtersize.item()+1)
    if gpu is not None: x = x.to(gpu); std = std.to(gpu)
    x = torch.meshgrid(x, x, indexing='ij')

    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    g = g / torch.sum(g)  # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0]/(std**3*2*math.pi)*torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1]/(std**3*2*math.pi)*torch.exp(-(x[1]/std)**2/2)*torch.exp(-(x[0]/std)**2/2)
    dgdy = dgdy / torch.sum(torch.abs(dgdy))  # Normalize

    # Stack and expand dim
    basis_filter = torch.stack([g,dgdx,dgdy], dim=0)[:,None,:,:]

    return basis_filter

#该函数用于对输入的batch进行高斯滤波卷积操作。
def convolve_gaussian_filters(batch, scale):
    if torch.isnan(scale).any():
        raise ValueError("scale为nan值")

    E, El, Ell = torch.split(batch, 1, dim=1)
    E_out, El_out, Ell_out = [], [], []

    for s in range(len(scale)):
        # Convolve with Gaussian filters
        w = gaussian_basis_filters(scale=scale[s:s+1], gpu=batch.device).to(dtype=batch.dtype)  # KCHW

        # the padding here works as "same" for odd kernel sizes
        E_out.append(F.conv2d(input=E[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        El_out.append(F.conv2d(input=El[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        Ell_out.append(F.conv2d(input=Ell[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))

    return torch.cat(E_out), torch.cat(El_out), torch.cat(Ell_out)



# == Color invariant definitions ==


def hat_H(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    H_single = torch.atan(El / (Ell + eps)) 
    return H_single


def hat_S(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    return (El ** 2 + Ell ** 2) / (E ** 2 + eps)


def hat_Ww(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wx = Ex / (E + eps)
    Wy = Ey / (E + eps)
    return Wx ** 2 + Wy ** 2


def hat_Wlw2(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wlx = Elx / (E + eps)
    Wly = Ely / (E + eps)
    return Wlx ** 2 + Wly ** 2


def hat_Wllw2(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wllx = Ellx / (E + eps)
    Wlly = Elly / (E + eps)
    return Wllx ** 2 + Wlly ** 2


# == Color invariant convolution ==


class PriorConv2d(nn.Module):
    def __init__(self, n_fea_middle, k=3, scale=0.0, ablation=None):

        super(PriorConv2d, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.n_fea_middle = n_fea_middle
        self.ablation = ablation
        self.single_prior_indices = {
            'only_H': 0,
            'only_S': 1,
            'only_RGB': 2,
            'only_O': 2,
            'only_Ww': 3,
            'only_W': 3,
        }
        self.single_prior_index = self.single_prior_indices.get(ablation)

        # Constants
        # RGB-order-only ablations do not use the GCM.  Other modes retain the
        # learnable, physics-initialized color basis.
        if self.single_prior_index != 2:
            self.gcm = torch.nn.Parameter(torch.tensor(
                [[0.06, 0.63, 0.27], [0.3, 0.04, -0.35],
                 [0.34, -0.6, 0.17]]))
        self.k = k

        # A Table-6 single-prior run instantiates just one projection branch
        # and a matching adapter.  This makes it a structural ablation rather
        # than a full four-branch model with three zero-weighted outputs.
        base_channels = n_fea_middle // 4
        active_indices = range(4) if self.single_prior_index is None else [self.single_prior_index]
        self.conv_H = (nn.Conv2d(1, base_channels, kernel_size=1, bias=True)
                       if 0 in active_indices else None)
        self.conv_S = (nn.Conv2d(1, base_channels, kernel_size=1, bias=True)
                       if 1 in active_indices else None)
        self.conv_RGB = (nn.Conv2d(3, base_channels, kernel_size=1, bias=True)
                         if 2 in active_indices else None)
        self.conv_Ww = (nn.Conv2d(1, base_channels, kernel_size=1, bias=True)
                        if 3 in active_indices else None)
        adapter_inputs = base_channels * (4 if self.single_prior_index is None else 1)
        self.conv_adjust = nn.Conv2d(
            adapter_inputs, n_fea_middle, kernel_size=1, bias=True)

        self.conv = None
        if self.single_prior_index != 2:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                nn.SiLU(),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                nn.SiLU(),
                torch.nn.Conv2d(16, 1, 3, padding=1)
            )
        
        #权重预测分支.引入一个动态权重生成模块，根据输入图像的特征预测每个先验的权重，使权重随着图像内容变化
        self.weight_predictor = None
        if self.single_prior_index is None and ablation not in ('equal_prior', 'equal'):
            self.weight_predictor = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 提取局部特征
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 进一步提取特征
                nn.ReLU(),
                nn.Conv2d(16, 4, kernel_size=1),  # 输出4个权重 logits
            )
        self.saved_features = {}

    def forward(self, x):
        # Make sure scale does not explode: clamp to max abs value of 2.5
        # self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        RGB_order = None
        if self.single_prior_index in (None, 2):
            with torch.no_grad():
                max_RGB = torch.argmax(x, dim=1)
                min_RGB = torch.argmin(x, dim=1)
                x_ = torch.flip(x, dims=(1,))
                max_RGB_ = 2 - torch.argmax(x_, dim=1)
                min_RGB_ = 2 - torch.argmin(x_, dim=1)

                def channel_one_hot(indices):
                    return F.one_hot(indices, num_classes=3).permute(0, 3, 1, 2).to(x.dtype)

                RGB_order = 0.5 * (channel_one_hot(max_RGB) + channel_one_hot(max_RGB_))
                RGB_order -= 0.5 * (channel_one_hot(min_RGB) + channel_one_hot(min_RGB_))

        H = S = Ww = scale = None
        weight_input = x
        if self.single_prior_index != 2:
            scale = torch.mean(self.conv(x), dim=(1, 2, 3))
            scale = torch.clamp(scale, min=-2.5, max=2.5)

            # Measure E, El, Ell by the Gaussian color model only when a
            # color-invariant prior is active.
            in_shape = x.shape
            gcm_input = x.view((in_shape[:2] + (-1,)))
            gcm_input = torch.matmul(
                self.gcm.to(x.device, dtype=x.dtype), gcm_input)
            gcm_input = gcm_input.view(
                (in_shape[0],) + (3,) + in_shape[2:])
            # Preserve the original FULL/no_* behavior: the dynamic fusion
            # predictor consumes the GCM-transformed channels, not raw RGB.
            weight_input = gcm_input
            E_out, El_out, Ell_out = convolve_gaussian_filters(
                gcm_input.float(), scale.float())
            E, Ex, Ey = torch.split(E_out, 1, dim=1)
            El = torch.split(El_out, 1, dim=1)[0]
            Ell = torch.split(Ell_out, 1, dim=1)[0]
            if self.single_prior_index in (None, 0):
                H = hat_H(E, Ex, Ey, El, None, None, Ell, None, None)
            if self.single_prior_index in (None, 1):
                S = torch.log(hat_S(E, Ex, Ey, El, None, None, Ell, None, None) + eps)
            if self.single_prior_index in (None, 3):
                Ww = torch.atan(hat_Ww(E, Ex, Ey, El, None, None, Ell, None, None))

        # 计算动态权重。单先验消融时使用固定权重 1，与论文的消融设定一致。
        weight_logits = (self.weight_predictor(weight_input)
                          if self.weight_predictor is not None else None)
        if self.single_prior_index is not None:
            weights = x.new_zeros((x.shape[0], 4, x.shape[2], x.shape[3]))
            weights[:, self.single_prior_index:self.single_prior_index + 1] = 1.0
        elif self.ablation in ('equal_prior', 'equal'):
            weights = x.new_full((x.shape[0], 4, x.shape[2], x.shape[3]), 0.25)
        else:
            prior_index = {'no_H': 0, 'no_S': 1, 'no_RGB': 2, 'no_O': 2,
                           'no_Ww': 3, 'no_W': 3}.get(self.ablation)
            if prior_index is not None:
                mask = torch.ones_like(weight_logits, dtype=torch.bool)
                mask[:, prior_index:prior_index + 1] = False
                weight_logits = weight_logits.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(weight_logits, dim=1)

        H_fea = S_fea = RGB_fea = Ww_fea = None
        H_fea_w = S_fea_w = RGB_fea_w = Ww_fea_w = None
        if self.single_prior_index is not None:
            if self.single_prior_index == 0:
                H_fea = self.conv_H(H)
                H_fea_w = H_fea
                active_feature = H_fea
            elif self.single_prior_index == 1:
                S_fea = self.conv_S(S)
                S_fea_w = S_fea
                active_feature = S_fea
            elif self.single_prior_index == 2:
                RGB_fea = self.conv_RGB(RGB_order)
                RGB_fea_w = RGB_fea
                active_feature = RGB_fea
            else:
                Ww_fea = self.conv_Ww(Ww)
                Ww_fea_w = Ww_fea
                active_feature = Ww_fea
            features = self.conv_adjust(active_feature)
        else:
            H_fea = self.conv_H(H)
            S_fea = self.conv_S(S)
            RGB_fea = self.conv_RGB(RGB_order)
            Ww_fea = self.conv_Ww(Ww)
            H_fea_w = H_fea * weights[:, 0:1, :, :]
            S_fea_w = S_fea * weights[:, 1:2, :, :]
            RGB_fea_w = RGB_fea * weights[:, 2:3, :, :]
            Ww_fea_w = Ww_fea * weights[:, 3:4, :, :]
            features = torch.cat(
                [H_fea_w, S_fea_w, RGB_fea_w, Ww_fea_w], dim=1)
            features = self.conv_adjust(features)

        # 保存原始特征和权重
        self.saved_features = {
            'ablation': self.ablation,
            'H_fea': H_fea,
            'S_fea': S_fea,
            'RGB_fea': RGB_fea,
            'RGB_order':RGB_order,
            'Ww_fea': Ww_fea,
            'H_fea_w': H_fea_w,
            'S_fea_w': S_fea_w,
            'RGB_fea_w': RGB_fea_w,
            'Ww_fea_w': Ww_fea_w,
            'weights': weights,
            'scale': scale.detach() if scale is not None else None,
            'sigma': (torch.pow(2.0, scale.detach())
                      if scale is not None else None),
            'features': features  # 来自最后的拼接
        }
        return features




#QuadPriorformer架构
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    '''该函数用于生成截断正态分布的张量。'''
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    '''该函数用于生成截断正态分布的张量。
      type:(Tensor, float, float, float, float) -> Tensor'''
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    '''该函数用于初始化张量，根据指定的模式和分布计算方差，并使用相应的分布初始化张量。'''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    '''该函数 lecun_normal_ 使用 variance_scaling_ 函数对输入的张量进行初始化。
    具体来说，它使用 fan_in 模式和 truncated_normal 分布来计算方差，并根据计算结果初始化张量。'''
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    '''输入数据上应用层归一化（Layer Normalization），然后再调用传入的函数 fn 进行处理。'''
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    '''该函数 shift_back 用于对输入的四维张量进行列偏移操作。具体步骤如下：
        获取输入张量的形状，计算下采样率。
        根据步长和下采样率调整步长值。
        遍历每个通道，根据调整后的步长对每一行进行偏移。
        返回处理后的张量。'''
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


    
class QP_MSA(nn.Module):
    def __init__(self, dim, heads, dim_head=40, use_guidance=True):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        #self.rescale定义了一个可学习的参数 rescale，用于缩放多头注意力机制中的每个头。
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.use_guidance = use_guidance
        self.saved_attn = {}  # 新增：用于保存中间特征

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        #这段代码的功能是将输入的查询、键、值和注意力矩阵进行重排。
        # 具体来说，使用 rearrange 函数将这些张量从形状 (b, n, h * d) 转换为 (b, h, n, d)，
        # 其中 b 是批次大小，n 是序列长度，h 是头数，d 是每个头的维度。
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v_illu = v * illu_attn
        if self.use_guidance:
            v = v_illu
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1) #位置编码
        out = out_c + out_p

        # 保存中间特征到字典
        self.saved_attn = {
            'x_in': x_in,
            'q_inp': q_inp.detach(),   # [b, h*w, c]
            'k_inp': k_inp.detach(),
            'v_inp': v_inp.detach(),
            'QK_attn': attn.detach(),  # [b, heads, n, n]
            'v_illu': v_illu.detach(),# [b, heads, n, d]
            'attn_output': out.detach()# [b, h, w, c]
        }
        return out

class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.conv_gate = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.conv_value = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.conv_out = nn.Conv2d(dim * mult, dim, 1, 1, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        x = x.permute(0, 3, 1, 2)  # 从 [b, h, w, c] 转为 [b, c, h, w]
        gate = self.swish(self.conv_gate(x))  # 门控分支，使用 SiLU 激活
        value = self.conv_value(x)  # 值分支
        out = gate * value  # 门控机制：逐元素相乘
        out = self.conv_out(out)  # 投影回原始维度
        return out.permute(0, 2, 3, 1)  # 转回 [b, h, w, c]




class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, use_se=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        self.se = SEBlock(dim) if use_se else nn.Identity()
    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.se(self.net(x.permute(0, 3, 1, 2).contiguous()))
        #将输入张量 x 的维度顺序进行调整，并通过连续内存布局优化后传递给网络层 self.net 进行前向传播
        return out.permute(0, 2, 3, 1)


def get_freq_indices(method):
    """Return the frequency indices published in the official FcaNet implementation."""
    valid_methods = {
        f'{prefix}{count}'
        for prefix in ('top', 'bot', 'low')
        for count in (1, 2, 4, 8, 16, 32)
    }
    if method not in valid_methods:
        raise ValueError(f'Unsupported FcaNet frequency selection method: {method}')

    num_freq = int(method[3:])
    all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2,
                         4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
    all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2,
                         6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
    all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0,
                         1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5,
                         4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
    all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5,
                         6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
    all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1,
                         4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]

    if method.startswith('top'):
        return all_top_indices_x[:num_freq], all_top_indices_y[:num_freq]
    if method.startswith('low'):
        return all_low_indices_x[:num_freq], all_low_indices_y[:num_freq]
    return all_bot_indices_x[:num_freq], all_bot_indices_y[:num_freq]


class MultiSpectralDCTLayer(nn.Module):
    """Fixed 2D DCT filters with DQPGT's cyclic channel assignment.

    The official FcaNet implementation requires ``channels % num_freq == 0``
    and uses equal-width channel groups. DQPGT contains 40-channel feature
    levels while its canonical FCAN selects 4 frequencies, so equal-width
    groups are impossible at those levels. This adaptation assigns frequency
    ``channel % num_freq`` and therefore gives every channel one DCT basis,
    with per-frequency channel counts differing by at most one.
    """
    def __init__(self, height, width, mapper_x, mapper_y, channels):
        super().__init__()
        if len(mapper_x) != len(mapper_y):
            raise ValueError('DCT frequency index lists must have the same length.')
        if len(mapper_x) > channels:
            raise ValueError('The number of DCT frequencies cannot exceed channels.')
        self.num_freq = len(mapper_x)
        self.register_buffer(
            'weight', self._build_dct_filter(height, width, mapper_x, mapper_y, channels))

    @staticmethod
    def _dct_value(pos, freq, size):
        value = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        return value if freq == 0 else value * math.sqrt(2)

    @classmethod
    def _build_dct_filter(cls, height, width, mapper_x, mapper_y, channels):
        dct_filter = torch.zeros(channels, height, width)
        for channel in range(channels):
            i = channel % len(mapper_x)
            freq_x, freq_y = mapper_x[i], mapper_y[i]
            for pos_x in range(height):
                for pos_y in range(width):
                    dct_filter[channel, pos_x, pos_y] = (
                        cls._dct_value(pos_x, freq_x, height)
                        * cls._dct_value(pos_y, freq_y, width))
        return dct_filter

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f'FcaNet DCT input must be 4D, but got {x.ndim}D.')
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return torch.sum(x * weight, dim=(2, 3))


class FrequencyChannelAttention(nn.Module):
    """FcaNet multi-spectral channel attention adapted to DQPGT's BHWC features."""
    def __init__(self, channels, reduction=16, dct_size=7, freq_sel_method='top4'):
        super().__init__()
        if dct_size < 7:
            raise ValueError('FcaNet dct_size must be at least 7.')
        self.dct_size = dct_size
        self._diagnostics_enabled = False
        self._diagnostic_gate = None

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [freq * (dct_size // 7) for freq in mapper_x]
        mapper_y = [freq * (dct_size // 7) for freq in mapper_y]
        self.dct_layer = MultiSpectralDCTLayer(
            dct_size, dct_size, mapper_x, mapper_y, channels)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, h, w, c = x.shape
        feat = x.permute(0, 3, 1, 2).contiguous()
        if h != self.dct_size or w != self.dct_size:
            pooled = F.adaptive_avg_pool2d(feat, (self.dct_size, self.dct_size))
        else:
            pooled = feat
        weight = self.fc(self.dct_layer(pooled)).view(b, c, 1, 1)
        if self._diagnostics_enabled:
            self._diagnostic_gate = weight.detach()
        return (feat * weight.expand_as(feat)).permute(0, 2, 3, 1).contiguous()


class IGAB(nn.Module):
    def __init__(self, dim, heads, dim_head=40, num_blocks=2, use_attn_norm=True,
                 use_se=False, use_fcan=True, fcan_freq_sel_method='top4',
                 use_guidance=True, use_qp_msa=True):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.use_attn_norm = use_attn_norm
        self.use_fcan = use_fcan
        # Table-4 baseline removes the complete QP-MSA branch.  This is
        # deliberately different from ``no_prior``, which retains ordinary
        # attention but bypasses its prior-guided V modulation.
        self.use_qp_msa = use_qp_msa
        self.saved_attn_features = {}
        for _ in range(num_blocks):
            block = []
            if use_qp_msa:
                attn = QP_MSA(dim=dim, heads=heads, dim_head=dim_head,
                              use_guidance=use_guidance)
                if use_attn_norm:
                    attn = PreNorm(dim, attn)
                block.append(attn)
            block.append(PreNorm(dim, FeedForward(dim=dim, use_se=use_se)))
            if use_fcan:
                block.append(FrequencyChannelAttention(
                    dim, freq_sel_method=fcan_freq_sel_method))
            self.blocks.append(nn.ModuleList(block))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        for block in self.blocks:
            if self.use_qp_msa:
                attn, ff = block[0], block[1]
                x = attn(x, illu_fea.permute(0, 2, 3, 1)) + x
                attn_module = attn.fn if isinstance(attn, PreNorm) else attn
                self.saved_attn_features = attn_module.saved_attn
                fcan_index = 2
            else:
                ff = block[0]
                fcan_index = 1
            ff_out = ff(x)
            if self.use_fcan:
                ff_out = block[fcan_index](ff_out)
            x = ff_out + x  # FCAN is inside the feed-forward residual branch.
        out = x.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
        return out

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=40, level=2, num_blocks=[2, 4, 4],
                 use_attn_norm=True, use_se=False, use_fcan=True,
                 fcan_freq_sel_method='top4', use_guidance=True,
                 use_qp_msa=True):
        super(Denoiser, self).__init__()
        if use_se and use_fcan:
            raise ValueError(
                'SE and FCAN are mutually exclusive in the DQPGT ablation protocol.')
        self.dim = dim
        self.level = level

        # 输入投影
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # 编码器
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, 
                     heads=dim_level // dim, use_attn_norm=use_attn_norm, use_se=use_se,
                     use_fcan=use_fcan, fcan_freq_sel_method=fcan_freq_sel_method,
                     use_guidance=use_guidance, use_qp_msa=use_qp_msa),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # 瓶颈
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, 
                              num_blocks=num_blocks[-1], use_attn_norm=use_attn_norm,
                              use_se=use_se, use_fcan=use_fcan,
                              fcan_freq_sel_method=fcan_freq_sel_method,
                              use_guidance=use_guidance,
                              use_qp_msa=use_qp_msa)

        # 解码器
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, 
                                  padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], 
                     dim_head=dim, heads=(dim_level // 2) // dim,
                     use_attn_norm=use_attn_norm, use_se=use_se, use_fcan=use_fcan,
                     fcan_freq_sel_method=fcan_freq_sel_method,
                     use_guidance=use_guidance, use_qp_msa=use_qp_msa),
            ]))
            dim_level //= 2

        # 输出投影
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''该方法用于初始化模型中的权重。
        对于线性层（nn.Linear），使用截断正态分布初始化权重，并将偏置初始化为0；
        对于层归一化层（nn.LayerNorm），将偏置初始化为0，权重初始化为1。'''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            # 检查并上采样 illu_fea，使其与 fea 的空间尺寸一致
            if illu_fea.shape[2:] != fea.shape[2:]:
                illu_fea = F.interpolate(
                    illu_fea, size=fea.shape[2:], mode='bilinear', align_corners=False
                )
            fea = LeWinBlock(fea, illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class RetinexIlluminationEstimator(nn.Module):
    """Retinexformer-style F_lu estimator used only by ablation A1."""
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(4, n_feat, 1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_feat, n_feat, 5, padding=2, bias=True, groups=n_feat)
        self.saved_features = {}

    def forward(self, img):
        img_mean = img.mean(dim=1, keepdim=True)
        guidance = self.depth_conv(self.conv1(torch.cat([img, img_mean], dim=1)))
        self.saved_features = {'features': guidance, 'ablation': 'retinex'}
        return guidance


class NoPriorGenerator(nn.Module):
    """Parameter-free placeholder for A0; QP-MSA also bypasses V gating."""
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.saved_features = {'ablation': 'no_prior'}

    def forward(self, img):
        return img.new_zeros((img.shape[0], self.n_feat, img.shape[2], img.shape[3]))


class QuadPriorFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, ablation=None,
                 num_blocks=[1, 1, 1], use_attn_norm=True, use_se=False, use_fcan=True,
                 fcan_freq_sel_method='top4', use_qp_msa=True):
        super(QuadPriorFormer_Single_Stage, self).__init__()
        if ablation == 'retinex':
            self.PriorConv2d = RetinexIlluminationEstimator(n_feat)
        elif ablation == 'no_prior':
            self.PriorConv2d = NoPriorGenerator(n_feat)
        else:
            self.PriorConv2d = PriorConv2d(n_feat, ablation=ablation)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, 
                                level=level, num_blocks=num_blocks,
                                use_attn_norm=use_attn_norm, use_se=use_se,
                                use_fcan=use_fcan,
                                fcan_freq_sel_method=fcan_freq_sel_method,
                                use_guidance=ablation != 'no_prior',
                                use_qp_msa=use_qp_msa)
        self.test_mode = False  # 添加模式标志，分开测试和训练

    def forward(self, img):
        # img:        b,c=3,h,w
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        illu_fea = self.PriorConv2d(img)
        input_img = img
        output_img = self.denoiser(input_img, illu_fea)
        if self.test_mode:  # 仅在测试模式返回特征
            prior_features = self.PriorConv2d.saved_features
            return {
                'output': output_img,
                'weights': prior_features.get('weights'),
                'H_fea': prior_features.get('H_fea'),
                'S_fea': prior_features.get('S_fea'),
                'RGB_fea': prior_features.get('RGB_fea'),
                'RGB_order': prior_features.get('RGB_order'),
                'Ww_fea': prior_features.get('Ww_fea'),
                'H_fea_w': prior_features.get('H_fea_w'),
                'S_fea_w': prior_features.get('S_fea_w'),
                'RGB_fea_w': prior_features.get('RGB_fea_w'),
                'Ww_fea_w': prior_features.get('Ww_fea_w'),
                'features': illu_fea,
                'QP_MSA_features': self.denoiser.encoder_layers[0][0].saved_attn_features,  # 假设第一个encoder层
             }
       
        else:
            return output_img  # 训练模式返回张量
        
class QuadPriorFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=3,
                 num_blocks=[1, 1, 1], ablation=None, use_attn_norm=True,
                 use_se=False, use_fcan=True, fcan_freq_sel_method='top4',
                 use_qp_msa=True):
        super(QuadPriorFormer, self).__init__()
        self.stage = stage
        modules_body = [QuadPriorFormer_Single_Stage(in_channels=in_channels, 
                                                  out_channels=out_channels, 
                                                  n_feat=n_feat, level=2, 
                                                  num_blocks=num_blocks, ablation=ablation,
                                                   use_attn_norm=use_attn_norm, use_se=use_se,
                                                   use_fcan=use_fcan,
                                                   fcan_freq_sel_method=fcan_freq_sel_method,
                                                   use_qp_msa=use_qp_msa,
                                                   )
                        for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        
    def set_test_mode(self, mode=True):
        """设置测试模式"""
        for module in self.body:
            if isinstance(module, QuadPriorFormer_Single_Stage):
                module.test_mode = mode
                
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if self.body[0].test_mode:  # 测试模式
            outputs = []
            for stage in self.body:
                stage_out = stage(x)
                outputs.append(stage_out)
                x = stage_out['output']  # 传递output到下一阶段
            return {
                'final_output': outputs[-1]['output'],
                'stage_outputs': outputs
            }
        else:  # 训练模式
            return self.body(x)  # 正常前向传播
