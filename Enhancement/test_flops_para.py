from utils import my_summary
from basicsr.models.archs.QuadpriorFormer_arch import QuadPriorFormer

my_summary(
    QuadPriorFormer(
        stage=1, num_blocks=[1, 2, 2], ablation=None,
        use_attn_norm=True, use_se=False, use_fcan=True,
        fcan_freq_sel_method='top4'),
    256, 256, 3, 1)
