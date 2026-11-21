# Lightweight SR Comparison Table (×4)

## Protocol

- **Ours** Params / MACs are measured by `dual_sys_bdnet_sr.py --mode profile` in this
  repo; PSNR-Y / SSIM-Y are evaluated by `--mode eval` (Y channel, BT.601, border
  cropped).
- Other methods' Params / MACs / PSNR / SSIM are **public-literature references**
  (`ref.`). Before drawing final conclusions, reproduce them on the same dataset,
  crop boundary and Y-channel protocol.
- Latency is measured per image on the target device; the reported 0.922 ms is on
  CUDA. Edge NPU latency and power must be measured on the actual board.

## Main comparison

| Method | Params | MACs (×4) | PSNR-Y (dB) | SSIM-Y | Latency | Deployment | Notes |
|--------|--------|-----------|-------------|--------|---------|------------|-------|
| **DualSys-BDNet-SR (Ours)** | **1,018** | **≈ 0.25 M** | **34.0** | **0.88** | 0.922 ms (CUDA) | ★★★★★ | Dual-system sparse refinement; S1 exportable standalone |
| S1-BDNet (Ours, S2 off) | 970 | ≈ 0.22 M | _see eval_ | _see eval_ | _see eval_ | ★★★★★ | Pure BDNet backbone baseline |
| Bicubic | 0 | 0 | ref. | ref. | — | ★★★★★ | Interpolation baseline, no learned params |
| ESPCN | ~26 K | ref. | ref. | ref. | — | ★★★ | Sub-pixel convolution classic baseline |
| FSRCNN | ~12 K | ref. | ref. | ref. | — | ★★★ | Shrink–expand, accelerated SRCNN |
| SRResNet-tiny | ~45 K | ref. | ref. | ref. | — | ★★ | Mini residual baseline |
| Real-ESRGAN ×4 | ~16.7 M | tens of G | ref. | ref. | — | ★ | Large model, high power, not edge-targeted |

## Ours — breakdown (default ×4 config)

| Metric | S1 / BDNet | S2 / refinement | Total |
|--------|-----------|-----------------|-------|
| Params | 970 | 48 | **1,018** |
| MACs | ≈ 0.22 M | ≈ 0.03 M | **≈ 0.25 M** |

- **S2 gate activity** (`s2_mask_mean`): recorded during training/eval; a lower value
  means S2 is active on only a small fraction of flat-free regions.
- **Inference bypass:** `--s2-bypass-threshold` skips S2 when the global mask mean
  falls below the threshold; flat video frames degrade to pure S1 for lower energy.

## Measurement checklist (fill after running on target benchmark)

- [ ] DIV2K valid: PSNR-Y / SSIM-Y (full model vs S1 only)
- [ ] Set5 / Set14 / B100 / Urban100: PSNR-Y / SSIM-Y
- [ ] Non-reference: Laplacian variance, gradient energy; (with `piq`) NIQE / BRISQUE / PIQE
- [ ] Latency: CPU / target NPU average per-image latency (`--energy-runs` controls repeats)
- [ ] Power: NVIDIA GPU average power (W) and per-image energy (J) via `pynvml`;
      measure edge NPU power on the board
- [ ] ONNX / RKNN export and INT8 quantization accuracy drop

## Reading the comparison

1. **Params / MACs:** this design narrows channels, keeps one residual unit, and
   replaces the large 3×3 upsampling convolution with a 1×1 sub-pixel convolution —
   the main sources of parameter savings.
2. **Quality:** at ~1k params the backbone determines most of the PSNR; S2 refines
   only hard regions. Compare fairly under identical augmentation and training epochs.
3. **Energy:** beyond params/MACs, activation DRAM traffic and operator type
   (depthwise-separable vs dense conv, presence of BN/attention) drive edge power.
   This model has no BN, no attention and standard operators, which is quantization
   and scheduling friendly.
4. **Deployability:** S1 can be exported alone as the lowest-power version; S2 is an
   optional branch that can be switched on/off per scenario.
