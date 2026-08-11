import yaml
from collections import OrderedDict
from copy import deepcopy
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def _deep_update(base, overrides):
    """Recursively merge a variant into a copied base configuration."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if (key in merged and isinstance(merged[key], dict)
                and isinstance(value, dict)):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_ablation_registry(variants_path):
    with open(variants_path, mode='r', encoding='utf-8') as file:
        Loader, _ = ordered_yaml()
        registry = yaml.load(file, Loader=Loader)
    if not isinstance(registry, dict) or not isinstance(registry.get('variants'), dict):
        raise ValueError(f'Invalid ablation registry: {variants_path}')
    return registry


def _resolve_variant(variants, variant_name, stack=None):
    if variant_name not in variants:
        available = ', '.join(variants.keys())
        raise KeyError(f'Unknown ablation variant "{variant_name}". Available: {available}')
    stack = [] if stack is None else stack
    if variant_name in stack:
        raise ValueError('Cyclic ablation inheritance: ' + ' -> '.join(stack + [variant_name]))
    entry = variants[variant_name] or {}
    resolved = OrderedDict()
    parent = entry.get('extends')
    if parent:
        resolved = _resolve_variant(variants, parent, stack + [variant_name])['overrides']
    resolved = _deep_update(resolved, entry.get('overrides', {}))
    return {
        'name': variant_name,
        'description': entry.get('description', ''),
        'mode': entry.get('mode', 'train_test'),
        'overrides': resolved
    }


def parse(opt_path, is_train=True, variant=None, variants_path=None):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r', encoding='utf-8') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    selected_variant = variant or opt.get('ablation_variant')
    configured_registry = variants_path or opt.get('ablation_variants_file')
    if selected_variant:
        if not configured_registry:
            raise ValueError('An ablation variant was selected but no registry file was configured.')
        if not osp.isabs(configured_registry):
            configured_registry = osp.join(osp.dirname(osp.abspath(opt_path)),
                                           configured_registry)
        registry = load_ablation_registry(configured_registry)
        resolved_variant = _resolve_variant(registry['variants'], selected_variant)
        if is_train and resolved_variant['mode'] == 'test_only':
            raise ValueError(f'Variant {selected_variant} is test-only and cannot be trained.')
        opt = _deep_update(opt, resolved_variant['overrides'])
        opt['ablation_variant'] = selected_variant
        opt['ablation_description'] = resolved_variant['description']
        opt['ablation_variant_mode'] = resolved_variant['mode']
        opt['_variants_path'] = osp.abspath(configured_registry)

    opt['is_train'] = is_train

    opt['name'] = osp.basename(opt_path).split('.')[0]
    if selected_variant:
        opt['name'] = f'{opt["name"]}_{selected_variant}'
    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])
        if dataset.get('split_manifest') is not None:
            dataset['split_manifest'] = osp.abspath(
                osp.expanduser(dataset['split_manifest']))

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        # Allow checkpoints/logs to live on a different, healthy filesystem.
        # This is especially useful when the repository or dataset is on a
        # removable/mounted data volume that is unsafe for frequent writes.
        configured_experiments_root = opt['path'].get('experiments_root')
        if configured_experiments_root:
            experiments_root = osp.abspath(
                osp.expanduser(configured_experiments_root))
        else:
            experiments_root = osp.join(opt['path']['root'], 'experiments',
                                        opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def save_options(opt, output_path):
    """Save the fully resolved runtime options for experiment provenance."""
    _, Dumper = ordered_yaml()
    with open(output_path, mode='w', encoding='utf-8') as file:
        yaml.dump(opt, file, Dumper=Dumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
