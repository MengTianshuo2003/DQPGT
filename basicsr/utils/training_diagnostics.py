import json
import math
import os

import torch


class TrainingDiagnostics:
    """Low-overhead diagnostics for DQPGT ablation experiments.

    Forward statistics are collected only on configured sampling iterations.
    Non-finite gradient parameters are inspected only after a non-finite global
    gradient norm is detected. Records are written as one JSON object per line.
    """

    def __init__(self, net, opt):
        cfg = opt.get('train', {}).get('diagnostics', {}) or {}
        self.enabled = bool(cfg.get('enabled', False)) and opt.get('rank', 0) == 0
        self.log_freq = max(int(cfg.get('log_freq', 1000)), 1)
        self.log_first_iter = bool(cfg.get('log_first_iter', True))
        self.track_fcan = bool(cfg.get('track_fcan', True))
        self.track_layernorm = bool(cfg.get('track_layernorm', True))
        self.track_grad_norms = bool(cfg.get('track_grad_norms', True))
        self.track_nonfinite = bool(cfg.get('track_nonfinite', True))
        self.track_prior = bool(cfg.get('track_prior', True))
        self.grad_module_types = set(cfg.get(
            'grad_module_types',
            ['PriorConv2d', 'IGAB', 'QP_MSA', 'FeedForward',
             'FrequencyChannelAttention', 'LayerNorm']))
        self.logger = None
        self.active = False
        self.current_iter = 0
        self.record = None
        self.handles = []

        output = cfg.get('output', 'diagnostics.jsonl')
        if not os.path.isabs(output):
            output = os.path.join(opt['path']['log'], output)
        self.output_path = output

        if not self.enabled:
            return

        # Import lazily to avoid a circular import through basicsr.utils.
        from basicsr.utils.logger import get_root_logger
        self.logger = get_root_logger()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._register_hooks(net)
        self.logger.info(
            f'Training diagnostics enabled: {self.output_path} '
            f'(log_freq={self.log_freq})')

    @staticmethod
    def _first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = TrainingDiagnostics._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = TrainingDiagnostics._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _rms(tensor):
        tensor = tensor.detach().float()
        return math.sqrt(torch.mean(tensor * tensor).item())

    @staticmethod
    def _stats(tensor):
        tensor = tensor.detach().float()
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std(unbiased=False).item(),
            'min': tensor.min().item(),
            'max': tensor.max().item()
        }

    @staticmethod
    def _layernorm_variance(tensor, normalized_shape):
        tensor = tensor.detach().float()
        dims = tuple(range(tensor.ndim - len(normalized_shape), tensor.ndim))
        return tensor.var(dim=dims, unbiased=False).mean().item()

    def _register_hooks(self, net):
        for name, module in net.named_modules():
            class_name = module.__class__.__name__
            if self.track_fcan and class_name == 'FrequencyChannelAttention':
                module._diagnostics_enabled = True
                self.handles.append(module.register_forward_hook(
                    self._make_fcan_hook(name)))
            if self.track_layernorm and isinstance(module, torch.nn.LayerNorm):
                self.handles.append(module.register_forward_hook(
                    self._make_layernorm_hook(name)))
            if self.track_prior and class_name == 'PriorConv2d':
                self.handles.append(module.register_forward_hook(
                    self._make_prior_hook(name)))

    def _make_prior_hook(self, name):
        initial_gcm = torch.tensor(
            [[0.06, 0.63, 0.27], [0.30, 0.04, -0.35], [0.34, -0.60, 0.17]])

        def hook(module, inputs, output):
            if not self.active:
                return
            saved = getattr(module, 'saved_features', {})
            gcm = module.gcm.detach().float().cpu()
            denom = torch.norm(initial_gcm).clamp_min(1e-12)
            entry = {
                'gcm': gcm.tolist(),
                'gcm_relative_drift': (
                    torch.norm(gcm - initial_gcm) / denom).item()
            }
            if 'scale' in saved:
                entry['scale'] = self._stats(saved['scale'])
            if 'sigma' in saved:
                entry['sigma'] = self._stats(saved['sigma'])
            if 'weights' in saved:
                entry['prior_weights'] = self._stats(saved['weights'])
            self.record['prior'][name] = entry
        return hook

    def _make_fcan_hook(self, name):
        def hook(module, inputs, output):
            if not self.active:
                return
            input_tensor = self._first_tensor(inputs)
            output_tensor = self._first_tensor(output)
            gate = getattr(module, '_diagnostic_gate', None)
            if input_tensor is None or output_tensor is None or gate is None:
                return
            input_rms = self._rms(input_tensor)
            output_rms = self._rms(output_tensor)
            self.record['fcan'][name] = {
                'gate': self._stats(gate),
                'activation_rms_before': input_rms,
                'activation_rms_after': output_rms,
                'activation_rms_ratio': output_rms / max(input_rms, 1e-12)
            }
        return hook

    def _make_layernorm_hook(self, name):
        def hook(module, inputs, output):
            if not self.active:
                return
            input_tensor = self._first_tensor(inputs)
            output_tensor = self._first_tensor(output)
            if input_tensor is None or output_tensor is None:
                return
            self.record['layernorm'][name] = {
                'variance_before': self._layernorm_variance(
                    input_tensor, module.normalized_shape),
                'variance_after': self._layernorm_variance(
                    output_tensor, module.normalized_shape)
            }
        return hook

    def begin_iteration(self, current_iter):
        if not self.enabled:
            return
        self.current_iter = current_iter
        self.active = (current_iter % self.log_freq == 0 or
                       (self.log_first_iter and current_iter == 1))
        self.record = {
            'iter': current_iter,
            'event': 'periodic',
            'fcan': {},
            'layernorm': {},
            'prior': {},
            'gradient_norms': {}
        } if self.active else None

    def collect_gradient_norms(self, net):
        if not (self.enabled and self.active and self.track_grad_norms):
            return
        for name, module in net.named_modules():
            if module.__class__.__name__ not in self.grad_module_types:
                continue
            squared_norm = 0.0
            parameter_count = 0
            has_gradient = False
            finite = True
            for parameter in module.parameters(recurse=True):
                if parameter.grad is None:
                    continue
                has_gradient = True
                gradient = parameter.grad.detach().float()
                parameter_count += gradient.numel()
                if not torch.isfinite(gradient).all().item():
                    finite = False
                else:
                    squared_norm += torch.sum(gradient * gradient).item()
            if has_gradient:
                self.record['gradient_norms'][name] = {
                    'l2': math.sqrt(squared_norm) if finite else None,
                    'finite': finite,
                    'parameter_count': parameter_count
                }

    def set_global_gradient_norm(self, grad_norm):
        if self.enabled and self.active:
            if torch.is_tensor(grad_norm):
                grad_norm = grad_norm.detach().float().item()
            self.record['global_gradient_norm_before_clip'] = float(grad_norm)

    def find_nonfinite_gradients(self, net):
        if not (self.enabled and self.track_nonfinite):
            return []
        affected = []
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all().item():
                affected.append({
                    'parameter': name,
                    'module': name.rsplit('.', 1)[0],
                    'nan_count': torch.isnan(gradient).sum().item(),
                    'posinf_count': torch.isposinf(gradient).sum().item(),
                    'neginf_count': torch.isneginf(gradient).sum().item()
                })
        return affected

    def report_nonfinite(self, event, details=None):
        if not self.enabled:
            return
        record = self.record if self.active and self.record is not None else {
            'iter': self.current_iter,
            'fcan': {},
            'layernorm': {},
            'prior': {},
            'gradient_norms': {}
        }
        record['event'] = event
        if details is not None:
            record['nonfinite'] = details
        self._write(record)
        self.logger.warning(
            f'[Diagnostics] iter={self.current_iter}, event={event}, '
            f'details={len(details) if isinstance(details, list) else details}')
        self.active = False
        self.record = None

    def finish_iteration(self):
        if not (self.enabled and self.active and self.record is not None):
            return
        self._write(self.record)
        self._log_summary(self.record)
        self.active = False
        self.record = None

    def record_validation_probe(self, net, current_iter, image_name):
        """Persist GCM/sigma state on the fixed first validation image."""
        if not (self.enabled and self.track_prior):
            return
        bare_net = net.module if hasattr(net, 'module') else net
        initial_gcm = torch.tensor(
            [[0.06, 0.63, 0.27], [0.30, 0.04, -0.35], [0.34, -0.60, 0.17]])
        record = {
            'iter': current_iter,
            'event': 'validation_probe',
            'image': image_name,
            'prior': {}
        }
        for name, module in bare_net.named_modules():
            if module.__class__.__name__ != 'PriorConv2d':
                continue
            saved = getattr(module, 'saved_features', {})
            gcm = module.gcm.detach().float().cpu()
            denom = torch.norm(initial_gcm).clamp_min(1e-12)
            entry = {
                'gcm': gcm.tolist(),
                'gcm_relative_drift': (
                    torch.norm(gcm - initial_gcm) / denom).item()
            }
            if 'scale' in saved:
                entry['scale'] = self._stats(saved['scale'])
            if 'sigma' in saved:
                entry['sigma'] = self._stats(saved['sigma'])
            if 'weights' in saved:
                entry['prior_weights'] = self._stats(saved['weights'])
            record['prior'][name] = entry
        self._write(record)

    def _write(self, record):
        with open(self.output_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(
                self._json_safe(record), ensure_ascii=False,
                allow_nan=False) + '\n')

    @staticmethod
    def _json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: TrainingDiagnostics._json_safe(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [TrainingDiagnostics._json_safe(item) for item in value]
        return value

    def _log_summary(self, record):
        parts = [f"[Diagnostics] iter={record['iter']}"]
        fcan = list(record.get('fcan', {}).values())
        if fcan:
            parts.append(
                'FCAN gate_mean={:.4f}, gate_range=[{:.4f},{:.4f}], '
                'rms_ratio={:.4f}'.format(
                    sum(x['gate']['mean'] for x in fcan) / len(fcan),
                    min(x['gate']['min'] for x in fcan),
                    max(x['gate']['max'] for x in fcan),
                    sum(x['activation_rms_ratio'] for x in fcan) / len(fcan)))
        layernorm = list(record.get('layernorm', {}).values())
        if layernorm:
            parts.append('LN var={:.4e}->{:.4e}'.format(
                sum(x['variance_before'] for x in layernorm) / len(layernorm),
                sum(x['variance_after'] for x in layernorm) / len(layernorm)))
        if 'global_gradient_norm_before_clip' in record:
            parts.append('global_grad={:.4e}'.format(
                record['global_gradient_norm_before_clip']))
        self.logger.info(' | '.join(parts))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
