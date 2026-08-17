
# DQPGT

This is the official code for "DQPGT: Dynamic Quadruple Priors Guided Transformer for Low-light Image Enhancement."


## Abstract


In the task of low-light image enhancement, many Retinex-based methods assume smooth illumination that often conflicts with complex real-world lighting conditions. When this assumption is violated, illumination becomes more difficult to disentangle from reflectance, noise, and chromatic shifts, potentially introducing local errors into the estimated Retinex guidance. These errors can distort the derived reflectance and may lead to uneven exposure, amplified noise, and color distortion in the enhanced results. This observation motivates us to investigate complementary color- and structure-aware guidance. This paper proposes a Dynamic Quadruple Priors Guided Transformer (DQPGT) method based on the Kubelka-Munk theory. DQPGT employs a Dynamic Quadruple Priors Estimator (DQPE) to dynamically fuse four descriptors into an input-dependent, spatially varying prior feature. This feature provides guidance to the Corruption Restorer (CR), directing the multi-head self-attention mechanism in the Transformer to model different regions of the image, thereby enabling precise region-specific enhancement. FCAN is further incorporated as a complementary multi-frequency channel-response recalibration module. Experiments on multiple benchmark datasets demonstrate that DQPGT achieves competitive quantitative performance and visual quality. Downstream evaluations further indicate its potential utility for subsequent computer vision tasks.



### 1. Download the project.

Please run the following command to ensure that you deploy our project locally.

```python
git clone https://github.com/MengTianshuo2003/DQPGT.git
```

### 2. Create Environment


- Make Conda Environment
```
conda create -n DQPGT python=3.8.10
conda activate DQPGT
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```


### 3. Prepare Dataset
If `data` is empty，please download the dataset from [Baidu Netdisk](https://pan.baidu.com/s/1_0oAd-92GzPy9TB4cr9h-g?pwd=ntbp) or [Google Drive](https://drive.google.com/drive/project/1LoKjzu5oskkrWrnj5mts37YLB06dK2__?usp=sharing) and place the data file in the DQPGT folder.

**Note:** 
Please download the `text_list.txt` and then put it into the folder `data/SMID`

The final placement should be as follows:

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |--SID
    |    |    |--short_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--long_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--SMID
    |    |    |--SMID_LQ_np
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--SMID_Long_np
    |    |    |    |--text_list.txt
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...

```


### Reproduction architecture switches
The `Options/QuadPriorFormer_*.yml` files expose the following architecture switches:

- `ablation: ~` keeps all four priors H/S/O/W enabled.
- `ablation: only_H`, `only_S`, `only_O`, or `only_W` keeps only that prior and fixes its weight to 1. The aliases `only_RGB` and `only_Ww` are also accepted.
- `ablation: no_H`, `no_S`, `no_RGB`, or `no_Ww` retains the existing leave-one-prior-out ablations.
- `use_fcan: true` enables the FcaNet-style multi-spectral channel attention block inside the feed-forward residual branch.
- `fcan_freq_sel_method: top4` uses the Top-4 frequency indices. Eight groups are used because all DQPGT channel widths (40/80/160) must be divisible by the number of frequency groups.

The FCAN implementation is adapted from the official [cfzd/FcaNet](https://github.com/cfzd/FcaNet) implementation.

### 4. Testing

Please ensure that the `pretrained_weights` folder contains our pre-trained weights. 
We will release the pretrained weights after the paper is formally published.

```shell
# activate the environment
conda activate DQPGT

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# No-reference Datasets
# Each --input_dir must point directly to the folder containing the test images.
# These paths are supplied at test time and are not read from the YAML file.
python3 Enhancement/test_from_nomonitor.py --input_dir /path/to/DICM --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset DICM

python3 Enhancement/test_from_nomonitor.py --input_dir /path/to/LIME --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LIME

python3 Enhancement/test_from_nomonitor.py --input_dir /path/to/MEF --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset MEF

python3 Enhancement/test_from_nomonitor.py --input_dir /path/to/NPE --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset NPE

python3 Enhancement/test_from_nomonitor.py --input_dir /path/to/VV --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset VV
```

#### About `--GT_mean`
`GT_mean` is disabled by default in our model. But if you want to enable it, just add a `--GT_mean` action at the end of the above test command as

```
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1 --GT_mean
```

### Evaluating the Params and FLOPS of models
Please run `Enhancement/test_flops_para.py` to test the parameters (Params) and floating-point operations (FLOPS) of DQPGT.

### 5. Training
Please ensure that you have fully completed the environment setup and can correctly infer the parameters and floating points.

```shell
# activate the enviroment
conda activate DQPGT

# LOL-v1
python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v2_real.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml

# SID
python3 basicsr/train.py --opt Options/QuadPriorFormer_SID.yml

# SMID
python3 basicsr/train.py --opt Options/QuadPriorFormer_SMID.yml

```


### 6.Ablation
All ablation studies in this paper were conducted on the LOLv2-synthetic dataset.

```
# list all variants
python basicsr/train.py --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml \
  --list-variants
```

#### train A0

```
python basicsr/train.py \
  --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml \
  --variant A0
```

#### test A0 

```
python basicsr/test.py \
  --opt Options/QuadPriorFormer_ablation_test.yml \
  --variant A0 \
  --weights /path/to/A0/best_psnr_xx.xx_iter.pth
```


### 7.Acknowledgments

We thank the following article and the authors for their open-source codes.

```
@article{retinexformer,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Yuanhao Cai and Hao Bian and Jing Lin and Haoqian Wang and Radu Timofte and Yulun Zhang},
  journal={2023 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023},
  pages={12470-12479},
  url={https://api.semanticscholar.org/CorpusID:257496232}
}

@INPROCEEDINGS{PQP,
  author={Wang, Wenjing and Yang, Huan and Fu, Jianlong and Liu, Jiaying},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Zero-Reference Low-Light Enhancement via Physical Quadruple Priors}, 
  year={2024},
  volume={},
  number={},
  pages={26057-26066},
  keywords={Training;Limiting;Lighting;Diffusion models;Distortion;Robustness;Pattern recognition;Low-light enhancement;diffusion;zero-reference;low-level vision;image processing},
  doi={10.1109/CVPR52733.2024.02462}}

@INPROCEEDINGS{FCAN,
  author={Qin, Zequn and Zhang, Pengyi and Wu, Fei and Li, Xi},
  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={FcaNet: Frequency Channel Attention Networks}, 
  year={2021},
  volume={},
  number={},
  pages={763-772},
  keywords={Image segmentation;Computer vision;Codes;Frequency-domain analysis;Computational modeling;Object detection;Computational efficiency;Recognition and classification},
  doi={10.1109/ICCV48922.2021.00082}}
```

