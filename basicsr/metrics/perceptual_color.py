"""Float-RGB perceptual/color metrics for the unified ablation protocol."""

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab


def _to_rgb_hwc_float(image):
    if torch.is_tensor(image):
        image = image.detach().float().cpu()
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError('Metrics expect one image at a time.')
            image = image[0]
        image = image.permute(1, 2, 0).numpy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f'Expected RGB HWC image, got shape {image.shape}.')
    return np.clip(image, 0.0, 1.0)


def calculate_delta_e00(img1, img2, crop_border=0, **kwargs):
    """Mean per-pixel CIEDE2000 under skimage's sRGB-D65 Lab conversion."""
    del kwargs
    rgb1 = _to_rgb_hwc_float(img1)
    rgb2 = _to_rgb_hwc_float(img2)
    if crop_border:
        rgb1 = rgb1[crop_border:-crop_border, crop_border:-crop_border]
        rgb2 = rgb2[crop_border:-crop_border, crop_border:-crop_border]
    return float(deltaE_ciede2000(rgb2lab(rgb1), rgb2lab(rgb2)).mean())


_LPIPS_ALEX = None


def calculate_lpips(img1, img2, crop_border=0, **kwargs):
    """LPIPS with the fixed AlexNet backbone and inputs mapped to [-1, 1]."""
    del kwargs
    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            'LPIPS metric requires the `lpips` package. Install it in the '
            'experiment environment before running E/F.') from error
    global _LPIPS_ALEX
    if _LPIPS_ALEX is None:
        _LPIPS_ALEX = lpips.LPIPS(net='alex').eval()
    rgb1 = _to_rgb_hwc_float(img1)
    rgb2 = _to_rgb_hwc_float(img2)
    if crop_border:
        rgb1 = rgb1[crop_border:-crop_border, crop_border:-crop_border]
        rgb2 = rgb2[crop_border:-crop_border, crop_border:-crop_border]
    tensor1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    tensor2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(_LPIPS_ALEX(tensor1, tensor2).item())
