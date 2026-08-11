"""Deterministic linear-RGB perturbations used by experiment D2."""

import torch


D2_PERTURBATION_KINDS = (
    'exposure_0.5',
    'gaussian_poisson',
    'mixed_illumination',
)


def srgb_to_linear(image):
    image = image.clamp(0, 1)
    return torch.where(
        image <= 0.04045, image / 12.92,
        torch.pow((image + 0.055) / 1.055, 2.4))


def linear_to_srgb(image):
    image = image.clamp(0, 1)
    return torch.where(
        image <= 0.0031308, image * 12.92,
        1.055 * torch.pow(image, 1 / 2.4) - 0.055)


def apply_ablation_perturbation(image, kind, seed=100):
    """Perturb BCHW RGB [0,1] input; GT is intentionally not accepted."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError('Expected BCHW RGB input.')
    if kind not in D2_PERTURBATION_KINDS:
        raise ValueError(
            f'Unknown D2 perturbation: {kind}. Supported values: '
            f'{", ".join(D2_PERTURBATION_KINDS)}')
    linear = srgb_to_linear(image)
    if kind == 'exposure_0.5':
        perturbed = linear * 0.5
    elif kind == 'gaussian_poisson':
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(seed))
        shot = torch.poisson((30.0 * linear).clamp_min(0), generator=generator) / 30.0
        read = torch.randn(
            linear.shape, dtype=linear.dtype, device=linear.device,
            generator=generator) * 0.02
        perturbed = shot + read
    elif kind == 'mixed_illumination':
        width = image.shape[-1]
        alpha = torch.linspace(
            0, 1, width, dtype=linear.dtype, device=linear.device).view(1, 1, 1, width)
        warm = linear.new_tensor([1.15, 1.00, 0.85]).view(1, 3, 1, 1)
        cool = linear.new_tensor([0.85, 1.00, 1.15]).view(1, 3, 1, 1)
        perturbed = linear * (warm * (1 - alpha) + cool * alpha)
    return linear_to_srgb(perturbed.clamp(0, 1))


def representation_distance(clean_guidance, perturbed_guidance, eps=1e-8):
    """Normalized L1 distance d_P for one model's common guidance adapter."""
    numerator = (perturbed_guidance - clean_guidance).abs().sum()
    denominator = clean_guidance.abs().sum().clamp_min(eps)
    return numerator / denominator
