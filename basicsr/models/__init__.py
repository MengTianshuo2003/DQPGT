import importlib
from os import path as osp

from basicsr.utils import get_root_logger, scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__)) #获取当前Python文件的绝对路径，并提取其所在的目录路径，存储在变量 model_folder 中
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
] #从指定文件夹中筛选出所有以 _model.py 结尾的文件，并提取这些文件的基名（去掉路径和扩展名）

# import all the model modules
#遍历 model_filenames 列表中的每个文件名，
# 并使用 importlib.import_module 动态导入对应的模块，
# 最终将所有导入的模块存储在 _model_modules 列表中。
_model_modules = [
    importlib.import_module(f'basicsr.models.{file_name}')
    for file_name in model_filenames
]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
