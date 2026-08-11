import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
import torch


def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))


def _ssim(img1, img2, data_range):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Single-channel image.
        img2 (ndarray): Single-channel image.
        data_range (float): Value range of the input images (1 or 255).

    Returns:
        float: ssim result.
    """

    if img1.shape != img2.shape:
        raise ValueError(
            f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if img1.ndim != 2:
        raise ValueError(
            f'_ssim expects a 2D image, but got shape {img1.shape}.')
    if min(img1.shape) < 11:
        raise ValueError(
            'SSIM requires both image dimensions to be at least 11 pixels.')

    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _infer_data_range(img1, img2, data_range=None):
    """Resolve a common input range while rejecting mixed-range inputs."""
    if data_range is not None:
        data_range = float(data_range)
        if data_range <= 0:
            raise ValueError(f'data_range must be positive, got {data_range}.')
        return data_range

    def _range_for_image(img):
        if np.issubdtype(img.dtype, np.integer):
            return 255.
        min_value = float(np.min(img))
        max_value = float(np.max(img))
        if min_value >= -1e-6 and max_value <= 1. + 1e-6:
            return 1.
        if min_value >= -1e-6 and max_value <= 255. + 1e-6:
            return 255.
        raise ValueError(
            'SSIM expects images in [0, 1] or [0, 255]. '
            f'Observed range [{min_value}, {max_value}].')

    range1 = _range_for_image(img1)
    range2 = _range_for_image(img2)
    if range1 != range2:
        raise ValueError(
            f'Images use different value ranges: {range1} and {range2}.')
    return range1


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   data_range=None):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        img2 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        data_range (float | None): Explicit input value range. If None, infer
            1 or 255 from dtype and values. Default: None.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    data_range = _infer_data_range(img1, img2, data_range)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        if data_range != 255.:
            img1 = img1 * (255. / data_range)
            img2 = img2 * (255. / data_range)
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim(img1[..., 0], img2[..., 0], 255.)

    ssims = [
        _ssim(img1[..., channel], img2[..., channel], data_range)
        for channel in range(img1.shape[2])
    ]
    return float(np.mean(ssims))
