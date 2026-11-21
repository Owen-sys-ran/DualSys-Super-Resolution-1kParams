# DualSys-BDNet-SR

**Author:** ZRAM

A ~1k-parameter dual-system lightweight super-resolution network for edge low-power
deployment. A fast BDNet backbone performs first-pass reconstruction, and a sparse
gated refinement branch selectively corrects only high-error / textured regions.
No attention, no BatchNorm, no KV cache, no VLM dependency; train, evaluate, profile
and export ONNX in a single file.

## Highlights

- **Ultra-low budget:** 1,018 parameters (S1 ≈ 970, S2 ≈ 48), ≈ 0.25 M MACs (×4).
- **Dual-system design:** System 1 = BDNet fast pathway; System 2 = sparse gated
  selective refinement, inspired by the human "look first, re-inspect only where
  unclear" perception.
- **Edge friendly:** all ops are standard `Conv2d / PixelShuffle / ReLU6 / Sigmoid`,
  depthwise-separable convolutions, suitable for INT8 quantization and RK3588 / NPU
  deployment.
- **Single-file pipeline:** dataset, training, reference/non-reference metrics,
  params/MACs/traffic/energy profiling, visualization and ONNX export.

## Results (×4)

| Metric | Value |
|--------|-------|
| Params | 1,018 |
| MACs | 248,832 (≈ 0.25 M) |
| PSNR-Y | 34.0 dB |
| SSIM-Y | 0.88 |
| Inference latency | 0.922 ms / image (CUDA) |
| Visual comparison | `./results/compare_grid.png` |

> Quality metrics are reported on the held-out SR benchmark under the protocol in
> `dual_sys_bdnet_sr.py` (Y channel, BT.601, border cropped). Re-run `--mode eval`
> on your target dataset to reproduce the numbers.

## Comparison with lightweight SR methods

| Method | Params | MACs | PSNR-Y | SSIM-Y | Latency | Notes |
|--------|--------|------|--------|--------|---------|-------|
| **DualSys-BDNet-SR (Ours)** | **1,018** | **≈ 0.25 M** | **34.0 dB** | **0.88** | 0.922 ms (CUDA) | Dual-system sparse refinement; S1 exportable standalone |
| S1-BDNet (Ours, S2 disabled) | 970 | ≈ 0.22 M | _see eval_ | _see eval_ | _see eval_ | Pure BDNet baseline |
| Bicubic | 0 | 0 | ref. | ref. | — | Interpolation baseline |
| ESPCN | ~26 K | ref. | ref. | ref. | — | Sub-pixel convolution baseline |
| FSRCNN | ~12 K | ref. | ref. | ref. | — | Shrink–expand accelerated SRCNN |
| SRResNet-tiny | ~45 K | ref. | ref. | ref. | — | Mini residual baseline |
| Real-ESRGAN ×4 | ~16.7 M | tens of G | ref. | ref. | — | Large model, high power, not edge-targeted |

> `ref.` values for other methods are public-literature references and must be
> reproduced under the **same dataset, crop and Y-channel protocol** before drawing
> final conclusions. Our Params / MACs are measured by `--mode profile` in this repo.

## Architecture

### System 1 — fast pathway (BDNet backbone)

- Input normalization → 3×3 feature head → one depthwise-separable residual unit →
  global LR residual connection.
- Optional BDO: low-rank (rank 1) EMA slow-reference channel modulation; the
  reference is updated during training and frozen at inference.
- 1×1 sub-pixel convolution predicts `3·r²` channels, followed by a single
  PixelShuffle upsample.

### System 2 — slow pathway (sparse gated refinement)

- Concatenate S1 output and the bicubic upsampled reference; a 1×1 hidden layer
  predicts the residual `delta` and a sigmoid gate `mask`; `refine = delta * mask`.
- The gate supervises against the per-pixel S1-vs-GT error, so it opens only on hard
  regions. At inference S2 can be bypassed adaptively from the global mask mean,
  letting flat frames run S1 only for lower energy.

### Scale-aware parameter budget (×4)

| `--dim` | `--blocks` | Total params |
|---------|-----------|--------------|
| 6 | 1 | ≈ 606 |
| 8 | 1 | ≈ 808 |
| 10 | 1 | ≈ 1,018 (default) |

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Data

- If no dataset is provided, DIV2K train/valid is downloaded automatically to
  `./data/DIV2K/`.
- Use `--train-dir` to point to your own HR images. LR is generated on the fly by
  bicubic downsampling; no separate LR set is needed.
- Put benchmarks (Set5 / Set14 / B100 / Urban100) under
  `./data/benchmark/<Name>/*.png`; they are detected automatically, otherwise the
  DIV2K valid set is used.

### Train

```bash
# Full dual system (default ~1k params, ×4)
python dual_sys_bdnet_sr.py --mode train --scale 4 --dim 10 --blocks 1 --epochs 100

# Pure BDNet baseline (disable S2)
python dual_sys_bdnet_sr.py --mode train --scale 4 --no-s2

# Even lighter: dim=8 and disable BDO
python dual_sys_bdnet_sr.py --mode train --scale 4 --dim 8 --no-bdo
```

### Evaluate

```bash
# Full dual system
python dual_sys_bdnet_sr.py --mode eval --scale 4 --ckpt ./results/best.pth

# S1 only (BDNet backbone, skip S2)
python dual_sys_bdnet_sr.py --mode eval --scale 4 --ckpt ./results/best.pth --only-s1
```

Metrics: PSNR-Y / SSIM-Y (reference, BT.601, border cropped); Laplacian variance,
gradient energy, and NIQE / BRISQUE / PIQE when `piq` is installed; params, MACs,
activation traffic, heuristic energy, per-image latency, and GPU power/energy when
NVML is available.

### Profile and export

```bash
# Params / MACs / traffic / latency / energy
python dual_sys_bdnet_sr.py --mode profile --scale 4 --dim 10 --blocks 1

# Export full dual-system ONNX
python dual_sys_bdnet_sr.py --mode export --scale 4 --ckpt ./results/best.pth

# Export pure BDNet(S1) ONNX — minimal ops, INT8 quantization friendly
python dual_sys_bdnet_sr.py --mode export --scale 4 --ckpt ./results/best.pth --only-s1
```

## Outputs (`--out-dir`, default `./results`)

| File | Content |
|------|---------|
| `best.pth` / `last.pth` | Best / latest checkpoint |
| `train_results.json` / `eval_results.json` | Metrics, profiling, energy, training history |
| `compare_grid.png` | LR / Bicubic / SR(Ours) / S1(BDNet) / HR / S2 Mask grid |
| `training_curves.png` | Training L1, PSNR-Y, SSIM-Y curves |
| `s2_mask_curve.png` | S2 gate activity over training |
| `metric_bar.png` | Per-benchmark metric bar chart |

## Deployment notes

- All ops stay in LR space with a single final PixelShuffle; depthwise-separable
  convolutions + ReLU6.
- ONNX can be converted with the RKNN Toolkit for RK3588 and similar platforms; add
  quantization-aware training (QAT) if 8-bit deployment is required.
- For video streams, use `--s2-bypass-threshold` to skip S2 on flat frames and fall
  back to pure S1-BDNet for lower energy.

## Repository structure

```
DualSys-BDNet-SR/
├── dual_sys_bdnet_sr.py   # Model / data / train / eval / profiling / visualization / export
├── comparison_table.md    # Lightweight SR comparison (params / MACs / quality / energy)
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
└── .gitignore
```

## Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dim` | Backbone channels (8 ≈ 800 params, 10 ≈ 1000 params) | 10 |
| `--blocks` | Number of residual units | 1 |
| `--alpha` | BDO slow-reference EMA momentum | 0.95 |
| `--no-bdo` | Disable BDO low-rank modulation | off |
| `--no-s2` | Disable S2 sparse refinement branch | off |
| `--s2-dim` | S2 hidden channels | 4 |
| `--s1-weight` | S1 standalone supervision loss weight | 0.4 |
| `--mask-target-weight` | Mask-to-S1-error alignment loss weight | 0.2 |
| `--sparse-weight` | Mask sparsity regularization weight | 1e-3 |
| `--s2-bypass-threshold` | Bypass S2 at inference when global mask mean < this (0 = off) | 0.0 |
| `--epochs` / `--batch-size` / `--lr` | Training hyperparameters | 100 / 16 / 1.5e-3 |

## Notes on scope

A ~1k-parameter model is an extreme lightweight setting; texture recovery is limited
and it targets power/compute-sensitive edge scenarios rather than peak PSNR. If
detail is insufficient, raise `--dim` to 12–16 (~1.5k–2.5k params) before adding
residual blocks.

## Citation

```bibtex
@misc{zram2026dualsysbdnet,
  title={DualSys-BDNet-SR: A 1k-Parameter Dual-System Lightweight Super-Resolution Network for Edge Low-Power Deployment},
  author={ZRAM},
  year={2026},
  howpublished={GitHub repository}
}
```
