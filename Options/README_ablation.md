# DQPGT ablation
The base training configuration specifies the data split, number of training epochs, optimizer, evaluation metrics, and logging; for each ablation, only the differences are declared in
`ablation_variants.yml`. Do not copy the entire training YAML file.

## command

列出全部变体：

```bash
python basicsr/train.py \
  --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml \
  --list-variants
```

train：

```bash
python basicsr/train.py \
  --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml \
  --variant A0
```
or :

```bash
torchrun --nproc_per_node=2 basicsr/train.py \
  --launcher pytorch \
  --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml \
  --variant B2
```

test best-validation-PSNR checkpoint：

```bash
python basicsr/test.py \
  --opt Options/QuadPriorFormer_LOL_v2_synthetic_test.yml \
  --variant B2 \
  --weights /path/to/B2/best_psnr_xx.xx_iter.pth
```



## Variant Mapping

- A: `A0`–`A7`: prior-based, equal-weight/dynamic fusion, and leave-one-out.
- B: `B0` is the no-channel-attention baseline; `B_SE` is the SE-only replacement control;
  `B1`–`B3` are FCAN-only with K=1/4/16. SE and FCAN are mutually exclusive; FULL and B2
  are both FCAN-only top-4; B3 is retained as the top-16 control.
- C: `C_FULL` and `C_NO_W` reuse A3 and A7, respectively.
- D: `D1` reuses A3; D2 is not retrained and uses
  `basicsr/data/ablation_perturbations.py` to test the A1/A3 checkpoints.
- E: `E0`–`E2`, no perceptual loss, relu3_3, relu4_3.
- F: `F_S0`, `F0`–`F4`, single-factor scan of SSIM/perceptual weights; SSIM weights set to
  0/0.1/0.3/0.5; perceptual weights reuse the value of 0 from E0 and are scanned across 0.005/0.02/0.1.

`A3`, `B2`, `F1`, `C_FULL`, and `D1` all inherit from `FULL` and do not require retraining.
The current official baseline is fixed as FCAN top4 and E1’s `relu3_3`: F/A/B all inherit `FULL` from the registry.
All active dataset YAML files and network constructors are uniformly set to the FCAN default value of `top4`; explicit frequency-sensitive variants such as B1/top1 and
B3/top16 are not affected by the default value migration.

## Automatic Logging

The training directory name automatically appends the variant ID, for example,
`experiments/QuadPriorFormer_LOL_v2_synthetic_A0`. Each run saves:

- `options_base.yml`: Base configuration;
- `ablation_variants.yml`: Variant registry for the current run;
- `options_resolved.yml`: Actual execution configuration after inheritance and overrides;
- `run_manifest.json`: variant, description, seed, Git commit, Train/validation
  data root directories, GT-mean, and checkpoint selection rules, explicitly recording whether Test is included in the selection.

The training validation, `basicsr/test.py`, and the two independent test entry points all use `gt_mean: false` by default; `FULL` explicitly overrides this setting, and all inherited variants keep it disabled.
`Enhancement/test_from_dataset.py` and `Enhancement/test_from_nomonitor.py` default to
`GT_mean=False`; `--GT_mean` is retained solely as a diagnostic switch, and its results must not be included in the official main dataset. 
GT-mean only alters post-processing for evaluation; it does not affect training gradients. After switching back to the off state, candidate checkpoints must be re-evaluated
and the best PSNR must be re-selected; the “best” status from when GT-mean was enabled cannot be directly carried over.

The current PyTorch implementation on the server does not provide a strictly deterministic implementation of `adaptive_avg_pool2d_backward_cuda`,
so the configuration uses `deterministic: true` and `deterministic_warn_only: true`: the seed, worker,
and cuDNN determinism settings remain fixed, but the backward of this operator only guarantees uninterrupted training; it should not be claimed to provide
bit-for-bit reproducibility. All production variants must maintain consistency with this setting.
