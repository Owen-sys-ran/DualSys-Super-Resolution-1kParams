#!/usr/bin/env python3
# coding: utf-8
"""
DualSys-BDNet-SR
Author: ZRAM

A single-file end-to-end lightweight super-resolution pipeline: a BDNet backbone
with a brain-inspired dual-system selective refinement branch.

One command covers: data loading -> training -> full/non-reference metrics ->
efficiency/energy profiling -> visualization (with the S2 gate mask) -> ONNX export.
The inference side is a pure local CNN; no network access and no large-model dependency.

================================================================================
Idea: a dual-system split closer to human visual perception
================================================================================
System 1 (fast pathway / feedforward intuition)
  - The lightweight BDNet-SR backbone: depthwise-separable convolutions + short
    residual + a single final PixelShuffle, all computed in LR space. No attention,
    no KV cache; standard operators, NPU/INT8 friendly.
  - Keeps the native BDNet "slow reference" mechanism: ChannelLowRankBDO maintains an
    EMA channel-statistic reference `ref` and applies low-rank deviation modulation
    with delta = f - ref (ref updated in training, frozen at inference).

System 2 (slow pathway / selective re-inspection, cautious refinement)
  - Instead of recomputing globally, it mimics human vision: S1 gives the first-pass
    result, and only high-error / textured regions are locally corrected. On flat
    regions the gate mask approaches 0, so S2 adds almost no compute.
  - Input: S1 output + bicubic upsampled reference; Output: residual `delta` and a
    spatial gate `mask`.
  - During training, the per-pixel S1-vs-GT error supervises the gate so it learns to
    "open only on hard regions"; at inference S2 can be bypassed adaptively from the
    global mask mean (flat frames run S1 only, lower energy).

Engineering orientation (embedded / RK3588 friendly)
  - All ops in LR space, a single final PixelShuffle; depthwise-separable conv + ReLU6.
  - Standard Conv2d / PixelShuffle / ReLU6 / Sigmoid; no permute buffer, no custom op.
  - S1 can be exported standalone; S2 is an optional add-on branch.

Metrics
  Reference   : PSNR-Y, SSIM-Y (BT.601, border cropped).
  Non-ref.    : Laplacian variance, gradient energy, NIQE/BRISQUE/PIQE (skipped if piq missing).
  Efficiency  : params, MACs, activation DRAM traffic, heuristic energy, per-image
                latency, measured power/energy when NVML is available.

Usage:
  # Train (default x4, S1=BDNet + S2 sparse refinement)
  python dual_sys_bdnet_sr.py --mode train --scale 4 --dim 10 --blocks 1

  # Train pure BDNet (disable S2) as baseline
  python dual_sys_bdnet_sr.py --mode train --scale 4 --no-s2

  # Evaluate checkpoint (full dual system)
  python dual_sys_bdnet_sr.py --mode eval --scale 4 --ckpt results/best.pth

  # Evaluate S1 only (BDNet backbone, skip S2)
  python dual_sys_bdnet_sr.py --mode eval --scale 4 --ckpt results/best.pth --only-s1

  # Profile params / MACs / traffic / latency / energy
  python dual_sys_bdnet_sr.py --mode profile --scale 4

  # Export ONNX (full dual system); use --only-s1 for pure BDNet
  python dual_sys_bdnet_sr.py --mode export --scale 4 --ckpt results/best.pth
"""

import argparse
import glob
import json
import math
import os
import random
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

# Optional deps (graceful degradation if missing)
try:
    import piq  # type: ignore
    _HAS_PIQ = True
except Exception:
    _HAS_PIQ = False

try:
    from thop import profile as thop_profile  # type: ignore
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

import cv2  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
# BT.601 SR mean (RGB, [0,1]), common in lightweight SR repos.
RGB_MEAN = (0.4488, 0.4371, 0.4040)
# Heuristic energy per 32-bit access (pJ); relative traffic comparison only, not chip-level measurement.
PJ_PER_32B_ACCESS = 4.5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root: str, recursive: bool = False) -> List[str]:
    files: List[str] = []
    for e in EXTS:
        if recursive:
            files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
        else:
            files.extend(glob.glob(os.path.join(root, e)))
    return sorted(files)


def download_url(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, dst)


def extract_zip(zip_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    for sub in sorted(glob.glob(os.path.join(out_dir, "*"))):
        if os.path.isdir(sub) and list_images(sub):
            return sub
    return out_dir


def _find_image_dir(root: str) -> str:
    if list_images(root):
        return root
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(sub) and list_images(sub):
            return sub
    return root


def ensure_div2k(data_dir: str):
    """Download + extract DIV2K train/valid; return (train_dir, valid_dir). Skip if present."""
    div_dir = os.path.join(data_dir, "DIV2K")
    train_dir = os.path.join(div_dir, "train")
    valid_dir = os.path.join(div_dir, "valid")
    if not list_images(train_dir, recursive=True):
        z = os.path.join(div_dir, "DIV2K_train_HR.zip")
        download_url(DIV2K_TRAIN_URL, z)
        train_dir = extract_zip(z, train_dir)
    else:
        train_dir = _find_image_dir(train_dir)
    if not list_images(valid_dir, recursive=True):
        z = os.path.join(div_dir, "DIV2K_valid_HR.zip")
        download_url(DIV2K_VALID_URL, z)
        valid_dir = extract_zip(z, valid_dir)
    else:
        valid_dir = _find_image_dir(valid_dir)
    return train_dir, valid_dir


# ---------------------------------------------------------------------------
# Dataset: on-the-fly bicubic LR/HR pairs
# ---------------------------------------------------------------------------
class SRDataset(Dataset):
    """Generate LR on the fly by bicubic downsampling HR."""

    def __init__(self, root: str, scale: int, patch_size: int, train: bool,
                 recursive: bool = True):
        self.files = list_images(root, recursive=recursive)
        if not self.files:
            raise FileNotFoundError(f"no images found under {root}")
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def _bicubic_lr(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        pil = ToPILImage()(hr_tensor.clamp(0, 1))
        _, h, w = hr_tensor.shape
        lr = pil.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        return ToTensor()(lr)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        hr = Image.open(fp).convert("RGB")
        if self.train:
            ps = min(self.patch_size, hr.width, hr.height)
            x0 = random.randint(0, hr.width - ps)
            y0 = random.randint(0, hr.height - ps)
            hr = hr.crop((x0, y0, x0 + ps, y0 + ps))
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                hr = hr.rotate(90, expand=True)
        else:
            w, h = hr.size
            hr = hr.crop((0, 0, w - w % self.scale, h - h % self.scale))

        hr_t = self.to_tensor(hr)
        lr_t = self._bicubic_lr(hr_t)
        return lr_t, hr_t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """BT.601 luma, [0,1]; x: CxHxW."""
    r, g, b = x[0], x[1], x[2]
    y = 16.0 / 255.0 + (65.481 * r + 12.255 * g + 24.966 * b) / 255.0
    return y.clamp(0.0, 1.0)


def _crop_border(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def psnr_y(sr: torch.Tensor, hr: torch.Tensor, scale: int) -> float:
    ys, yh = rgb_to_y(sr), rgb_to_y(hr)
    ys, yh = _crop_border(ys, scale), _crop_border(yh, scale)
    mse = F.mse_loss(ys, yh).item()
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_y(sr: torch.Tensor, hr: torch.Tensor, scale: int) -> float:
    from skimage.metrics import structural_similarity as sk_ssim
    ys, yh = rgb_to_y(sr), rgb_to_y(hr)
    ys, yh = _crop_border(ys, scale), _crop_border(yh, scale)
    return float(sk_ssim(
        yh.numpy(), ys.numpy(), data_range=1.0,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    ))


def nr_metrics(sr: torch.Tensor) -> Dict[str, float]:
    """Non-reference: Laplacian variance, gradient energy; NIQE/BRISQUE/PIQE if piq available."""
    y = (rgb_to_y(sr).numpy() * 255.0).astype(np.uint8)
    lap = float(cv2.Laplacian(y, cv2.CV_64F).var())
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1)
    grad = float((gx ** 2 + gy ** 2).mean())
    out: Dict[str, float] = {"laplacian_var": lap, "gradient_energy": grad}

    if _HAS_PIQ:
        t = sr.unsqueeze(0).clamp(0, 1)
        try:
            out["niqe"] = float(piq.niqe(t).item())
        except Exception:
            pass
        try:
            out["brisque"] = float(piq.brisque(t).item())
        except Exception:
            pass
        try:
            out["piqe"] = float(piq.piqe(t).item())
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# System 1: ultra-light BDNet-SR backbone (~1k params, keeps BDO/EMA slow reference)
# ---------------------------------------------------------------------------
class ChannelLowRankBDO(nn.Module):
    """Low-rank baseline deformation channel modulation (BDNet Eq.1-3), fixed rank=1 to save params.

    delta = f - s ; corr = gamma * (V(W delta)) ; W: D->1, V: 1->D。
    1x1 conv, keeps NCHW, no permute buffer; reference s is a detached EMA buffer,
    updated in training only, frozen at inference.
    """

    def __init__(self, dim: int, alpha: float):
        super().__init__()
        self.w = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.v = nn.Conv2d(1, dim, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("ref", torch.zeros(1, dim, 1, 1))
        self.alpha = alpha

    @torch.no_grad()
    def update_reference(self, x: torch.Tensor) -> None:
        mu = x.mean(dim=(0, 2, 3), keepdim=True)
        self.ref.mul_(self.alpha).add_(mu.detach(), alpha=1.0 - self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = x - self.ref
        corr = self.v(self.w(delta))
        return corr * self.gamma.view(1, -1, 1, 1)


class MicroBlock(nn.Module):
    """Ultra-light residual unit: depthwise-separable 3x3 spatial mix + 1x1 channel mix, single short residual."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.act(self.dw(x)))


class MicroBDNetSR(nn.Module):
    """System 1 (fast pathway): ~1k-param BDNet-SR backbone, standalone inference / ONNX export.

    Key params savings:
      - narrow channels, one/few residual units;
      - 1x1 conv predicts 3*r² sub-pixel channels directly (no big 3x3 upsampling conv);
      - no separate tail/out 3x3 conv; sub-pixel layer outputs RGB residual directly.
    """

    def __init__(self, scale: int, dim: int = 10, blocks: int = 1,
                 alpha: float = 0.95, use_bdo: bool = True, in_chans: int = 3):
        super().__init__()
        self.scale = scale
        self.use_bdo = use_bdo
        self.register_buffer("mean", torch.tensor(RGB_MEAN).view(1, in_chans, 1, 1))
        self.head = nn.Conv2d(in_chans, dim, kernel_size=3, padding=1, bias=False)
        self.body = nn.ModuleList([MicroBlock(dim) for _ in range(blocks)])
        if use_bdo:
            self.bdo = ChannelLowRankBDO(dim, alpha)
        # 1x1 sub-pixel prediction: dim -> 3*r², PixelShuffle upsample
        self.subpixel = nn.Conv2d(dim, in_chans * scale * scale, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(scale)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        h = self.head(x)
        shortcut = h
        for blk in self.body:
            h = blk(h)
        if self.use_bdo:
            if self.training:
                self.bdo.update_reference(h)
            h = h + self.bdo(h)
        h = shortcut + h                       # single global LR residual
        out = self.shuffle(self.subpixel(h))    # single final sub-pixel upsample
        return out + self.mean


# ---------------------------------------------------------------------------
# System 2: ultra-light sparse-gated cautious refinement branch (selective re-inspection)
# ---------------------------------------------------------------------------
class System2Refine(nn.Module):
    """Slow pathway: 1x1 conv predicts residual delta and spatial gate mask.

    mask in (0,1): ~0 on flat regions (no correction, no compute), ~1 on texture/edge/error regions.
    All 1x1 conv, very few params; no attention, no recurrence.
    """

    def __init__(self, dim: int = 4, in_chans: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_chans * 2, dim, kernel_size=1, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.conv_delta = nn.Conv2d(dim, in_chans, kernel_size=1, bias=False)
        self.conv_mask = nn.Conv2d(dim, in_chans, kernel_size=1, bias=False)

    def forward(self, s1_hr: torch.Tensor, bicubic_hr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([s1_hr, bicubic_hr], dim=1)
        feat = self.act(self.conv_in(cat))
        delta = self.conv_delta(feat)
        mask = torch.sigmoid(self.conv_mask(feat))
        return delta, mask


class DualSysBDNetSR(nn.Module):
    """Dual-system fusion: S1 = ultra-light BDNet-SR fast pathway; S2 = sparse-gated selective refinement.

    forward returns (final_hr, s1_hr, mask):
      - final_hr = s1_hr + delta * mask (S2 enabled);
      - with S2 disabled, final_hr = s1_hr and mask is a zero placeholder.
    """

    def __init__(self, scale: int, dim: int = 10, blocks: int = 1,
                 alpha: float = 0.95, use_bdo: bool = True,
                 use_s2: bool = True, s2_dim: int = 4,
                 s2_bypass_threshold: float = 0.0, in_chans: int = 3):
        super().__init__()
        self.scale = scale
        self.use_s2 = use_s2
        self.s2_bypass_threshold = s2_bypass_threshold
        self.s1 = MicroBDNetSR(
            scale=scale, dim=dim, blocks=blocks, alpha=alpha,
            use_bdo=use_bdo, in_chans=in_chans,
        )
        if use_s2:
            self.s2 = System2Refine(dim=s2_dim, in_chans=in_chans)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1_hr = self.s1(x)
        if not self.use_s2:
            zero_mask = torch.zeros_like(s1_hr)
            return s1_hr, s1_hr, zero_mask

        bicubic = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                                align_corners=False)
        delta, mask = self.s2(s1_hr, bicubic)
        refine = delta * mask

        # Adaptive inference bypass: skip S2 when global mask mean is below threshold (pure S1 output).
        if (not self.training) and self.s2_bypass_threshold > 0.0:
            mask_mean = mask.mean().item()
            if mask_mean < self.s2_bypass_threshold:
                return s1_hr, s1_hr, mask

        final_hr = s1_hr + refine
        return final_hr, s1_hr, mask


# ---------------------------------------------------------------------------
# Efficiency profiling: params / MACs / activation traffic / latency / energy
# ---------------------------------------------------------------------------
def profile_model(model: nn.Module, lr: torch.Tensor) -> Dict:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = 0
    if _HAS_THOP:
        model.eval()
        macs, _ = thop_profile(model, inputs=(lr,), verbose=False)

    traffic = {"in_elems": 0, "out_elems": 0}
    hooks = []

    def hook(mod, inp, out):
        traffic["in_elems"] += inp[0].numel()
        traffic["out_elems"] += out.numel()

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(lr)
    for h in hooks:
        h.remove()

    bytes_elem = 4  # FP32; halves to 2 under AMP
    total_bytes = (traffic["in_elems"] + traffic["out_elems"]) * bytes_elem
    traffic_joule = total_bytes * PJ_PER_32B_ACCESS * 1e-12
    return {
        "params": int(params),
        "macs": int(macs),
        "activation_in_elems": int(traffic["in_elems"]),
        "activation_out_elems": int(traffic["out_elems"]),
        "traffic_bytes": int(total_bytes),
        "traffic_mb": round(total_bytes / 1e6, 3),
        "heuristic_traffic_uJ": round(traffic_joule * 1e6, 3),
        "bytes_per_elem": bytes_elem,
        "pJ_per_access": PJ_PER_32B_ACCESS,
    }


def benchmark_latency(model: nn.Module, lr: torch.Tensor,
                       n: int = 50, warmup: int = 10) -> Dict:
    device = lr.device
    model.eval()
    is_cuda = device.type == "cuda"

    def _run():
        with torch.no_grad():
            model(lr)

    for _ in range(warmup):
        _run()
    if is_cuda:
        torch.cuda.synchronize()

    nvml = None
    handle = None
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        nvml = pynvml
    except Exception:
        nvml = None

    powers: List[float] = []
    t0 = time.time()
    for _ in range(n):
        _run()
        if nvml is not None:
            powers.append(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)  # mW -> W
        if is_cuda:
            torch.cuda.synchronize()
    elapsed = time.time() - t0

    res: Dict = {
        "latency_ms_per_image": round(elapsed / n * 1000.0, 3),
        "device": str(device),
        "runs": n,
    }
    if nvml is not None and powers:
        avg_w = float(np.mean(powers))
        res["avg_power_w"] = round(avg_w, 3)
        res["energy_j_per_image"] = round(avg_w * elapsed / n, 6)
    return res


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _forward(model: nn.Module, lr: torch.Tensor, only_s1: bool
             ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return (sr, s1, mask); compatible with dual-system and S1-only models."""
    out = model(lr)
    if isinstance(out, tuple):
        final, s1, mask = out
        sr = s1 if only_s1 else final
        return sr, s1, mask
    return out, out, None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, scale: int, device: torch.device,
             max_items: Optional[int] = None, only_s1: bool = False) -> Dict:
    model.eval()
    psnrs, ssims = [], []
    s1_psnrs, s1_ssims = [], []
    nr_buf: Dict[str, List[float]] = {
        "laplacian_var": [], "gradient_energy": [],
        "niqe": [], "brisque": [], "piqe": [],
    }
    mask_means: List[float] = []
    counted = 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr, s1, mask = _forward(model, lr, only_s1)
        sr = sr.clamp(0, 1)
        s1 = s1.clamp(0, 1)
        if mask is not None and mask.numel() > 0:
            mask_means.append(float(mask.mean().item()))
        for i in range(sr.size(0)):
            sri, hri, s1i = sr[i].cpu(), hr[i].cpu(), s1[i].cpu()
            psnrs.append(psnr_y(sri, hri, scale))
            ssims.append(ssim_y(sri, hri, scale))
            s1_psnrs.append(psnr_y(s1i, hri, scale))
            s1_ssims.append(ssim_y(s1i, hri, scale))
            nr = nr_metrics(sri)
            for k in nr_buf:
                if k in nr:
                    nr_buf[k].append(nr[k])
            counted += 1
            if max_items and counted >= max_items:
                break
        if max_items and counted >= max_items:
            break

    out: Dict = {
        "psnr_y": round(float(np.mean(psnrs)), 3),
        "ssim_y": round(float(np.mean(ssims)), 4),
        "s1_psnr_y": round(float(np.mean(s1_psnrs)), 3),
        "s1_ssim_y": round(float(np.mean(s1_ssims)), 4),
        "n_images": len(psnrs),
    }
    if mask_means:
        out["s2_mask_mean"] = round(float(np.mean(mask_means)), 4)
    for k, v in nr_buf.items():
        if v:
            out[k] = round(float(np.mean(v)), 4)
    return out


def build_eval_loaders(args, valid_dir: str) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    bench_dir = os.path.join(args.data_dir, "benchmark")
    if os.path.isdir(bench_dir):
        for sub in sorted(glob.glob(os.path.join(bench_dir, "*"))):
            if os.path.isdir(sub) and list_images(sub, recursive=True):
                ds = SRDataset(sub, args.scale, args.patch_size, train=False)
                loaders[os.path.basename(sub)] = DataLoader(
                    ds, batch_size=1, shuffle=False, num_workers=args.workers)
    if not loaders:
        ds = SRDataset(valid_dir, args.scale, args.patch_size, train=False)
        loaders["DIV2K_valid"] = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=args.workers)
    return loaders


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _to_pil(t: torch.Tensor) -> Image.Image:
    return ToPILImage()(t.clamp(0, 1).cpu())


def _mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """mask: CxHxW -> grayscale PIL (channel mean)."""
    m = mask.mean(dim=0, keepdim=True).clamp(0, 1).cpu()
    return ToPILImage()(m)


def save_compare_grid(model: nn.Module, loader: DataLoader, scale: int,
                      device: torch.device, out_path: str, n: int = 4,
                      show_mask: bool = True) -> None:
    """Save LR / Bicubic / SR(Ours) / S1(BDNet) / HR / S2-mask comparison grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    rows = []
    rows_mask: List[Optional[Image.Image]] = []
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        with torch.no_grad():
            res = model(lr)
            if isinstance(res, tuple):
                sr, s1, mask = res
            else:
                sr, s1, mask = res, res, None
        bic = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
        for i in range(sr.size(0)):
            rows.append([_to_pil(lr[i]), _to_pil(bic[i]), _to_pil(sr[i]),
                         _to_pil(s1[i]), _to_pil(hr[i])])
            rows_mask.append(_mask_to_pil(mask[i]) if mask is not None else None)
        if len(rows) >= n:
            break
    rows = rows[:n]
    rows_mask = rows_mask[:n]

    cols = 5 + (1 if show_mask and any(m is not None for m in rows_mask) else 0)
    fig, axes = plt.subplots(len(rows), cols, figsize=(3 * cols, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    titles = ["LR", "Bicubic", "SR (S1+S2)", "S1 (BDNet)", "HR"]
    if cols == 6:
        titles.append("S2 Mask")
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c])
        if cols == 6:
            axes[r, 5].imshow(rows_mask[r], cmap="gray", vmin=0, vmax=1)
            axes[r, 5].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] compare grid -> {out_path}")


def plot_training_curves(history: List[Dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]

    bench_names = sorted({k for h in history for k in h if isinstance(h[k], dict)})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(epochs, losses, marker="o", ms=3)
    ax[0].set_title("Train L1 Loss"); ax[0].set_xlabel("epoch"); ax[0].grid(True)

    for name in bench_names:
        ps = [h[name]["psnr_y"] for h in history if name in h]
        ss = [h[name]["ssim_y"] for h in history if name in h]
        ep = [h["epoch"] for h in history if name in h]
        ax[1].plot(ep, ps, marker="o", ms=3, label=name)
        ax[2].plot(ep, ss, marker="o", ms=3, label=name)
    ax[1].set_title("PSNR-Y"); ax[1].set_xlabel("epoch"); ax[1].grid(True); ax[1].legend()
    ax[2].set_title("SSIM-Y"); ax[2].set_xlabel("epoch"); ax[2].grid(True); ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    nr_keys = ["niqe", "brisque", "piqe", "laplacian_var", "gradient_energy"]
    nr_present = [k for k in nr_keys if any(
        isinstance(h.get(b), dict) and k in h[b] for h in history for b in bench_names)]
    if nr_present and bench_names:
        fig, ax = plt.subplots(1, len(nr_present), figsize=(5 * len(nr_present), 4))
        if len(nr_present) == 1:
            ax = [ax]
        b0 = bench_names[0]
        for j, k in enumerate(nr_present):
            ys = [h[b0][k] for h in history if b0 in h and k in h[b0]]
            ep = [h["epoch"] for h in history if b0 in h and k in h[b0]]
            ax[j].plot(ep, ys, marker="o", ms=3)
            ax[j].set_title(k); ax[j].set_xlabel("epoch"); ax[j].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "nr_curves.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # S2 gate-mask activity curve
    mask_curve = [h["s2_mask_mean"] for h in history if "s2_mask_mean" in h]
    if mask_curve:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs[:len(mask_curve)], mask_curve, marker="o", ms=3, color="tab:orange")
        ax.set_title("S2 Gate Mask Mean"); ax.set_xlabel("epoch"); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "s2_mask_curve.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"[viz] training curves -> {out_dir}")


def plot_metric_bar(results: Dict[str, Dict], out_dir: str) -> None:
    """Bar chart of per-benchmark metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    metric_keys = ["psnr_y", "ssim_y", "s1_psnr_y", "s1_ssim_y",
                   "niqe", "brisque", "piqe", "laplacian_var", "gradient_energy"]
    present = [k for k in metric_keys if any(k in results[n] for n in names)]
    if not present:
        return
    fig, ax = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        ax = [ax]
    x = np.arange(len(names))
    for j, k in enumerate(present):
        vals = [results[n].get(k, 0.0) for n in names]
        ax[j].bar(x, vals)
        ax[j].set_title(k); ax[j].set_xticks(x); ax[j].set_xticklabels(names, rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_bar.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] metric bar -> {out_dir}")


# ---------------------------------------------------------------------------
# Loss: reconstruction + S1 standalone supervision + mask error alignment + sparsity
# ---------------------------------------------------------------------------
def mask_target_from_error(s1: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-image normalize |s1-gt| to [0,1] as the S2 gate supervision target (detached)."""
    err = (s1 - gt).abs().mean(dim=1, keepdim=True)  # Bx1xHxW
    b = err.size(0)
    tgt = torch.zeros_like(err)
    for i in range(b):
        e = err[i]
        emin, emax = e.min(), e.max()
        if (emax - emin) > 1e-6:
            tgt[i] = (e - emin) / (emax - emin)
        else:
            tgt[i] = 0.0
    return tgt.detach()


def compute_loss(model: nn.Module, sr: torch.Tensor, s1: torch.Tensor,
                 mask: torch.Tensor, gt: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    l1 = nn.L1Loss()
    loss_rec = l1(sr, gt)
    loss_s1 = l1(s1, gt)

    logs: Dict[str, float] = {
        "loss_rec": loss_rec.item(),
        "loss_s1": loss_s1.item(),
    }
    total = loss_rec + args.s1_weight * loss_s1

    if model.use_s2 and mask.numel() > 0:
        tgt = mask_target_from_error(s1, gt)
        mask_gray = mask.mean(dim=1, keepdim=True)
        loss_mask = F.l1_loss(mask_gray, tgt)
        loss_sparse = mask_gray.mean()
        total = total + args.mask_target_weight * loss_mask + args.sparse_weight * loss_sparse
        logs["loss_mask"] = loss_mask.item()
        logs["loss_sparse"] = loss_sparse.item()
        logs["mask_mean"] = float(mask_gray.mean().item())

    logs["loss_total"] = total.item()
    return total, logs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    if args.train_dir and os.path.isdir(args.train_dir):
        train_dir = args.train_dir
        _, valid_dir = ensure_div2k(args.data_dir)
    else:
        train_dir, valid_dir = ensure_div2k(args.data_dir)

    train_ds = SRDataset(train_dir, args.scale, args.patch_size, train=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda")
    eval_loaders = build_eval_loaders(args, valid_dir)

    model = DualSysBDNetSR(
        scale=args.scale, dim=args.dim, blocks=args.blocks,
        alpha=args.alpha, use_bdo=not args.no_bdo, use_s2=not args.no_s2,
        s2_dim=args.s2_dim, s2_bypass_threshold=args.s2_bypass_threshold,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    s1_params = sum(p.numel() for p in model.s1.parameters() if p.requires_grad)
    print(f"[train] trainable parameters: {n_params:,} "
          f"(S1/BDNet: {s1_params:,}, S2: {n_params - s1_params:,})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_score = -1e9
    history: List[Dict] = []
    ckpt_best = os.path.join(args.out_dir, "best.pth")
    ckpt_last = os.path.join(args.out_dir, "last.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                res = model(lr)
                if isinstance(res, tuple):
                    sr, s1, mask = res
                else:
                    sr, s1, mask = res, res, None
                loss, logs = compute_loss(model, sr, s1, mask, hr, args)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * lr.size(0)
            seen += lr.size(0)

        train_loss = loss_sum / seen
        record: Dict = {"epoch": epoch, "train_loss": round(train_loss, 4),
                        "lr": round(sched.get_last_lr()[0], 7),
                        "time_s": round(time.time() - t0, 1)}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            epoch_scores: Dict[str, float] = {}
            for name, loader in eval_loaders.items():
                m = evaluate(model, loader, args.scale, device, only_s1=False)
                record[name] = m
                epoch_scores[name] = m["psnr_y"]
                extra = f" S1-PSNR={m['s1_psnr_y']:.3f}"
                if "s2_mask_mean" in m:
                    extra += f" mask={m['s2_mask_mean']:.3f}"
                print(f"[train] epoch {epoch} {name}: "
                      f"PSNR={m['psnr_y']:.3f} SSIM={m['ssim_y']:.4f}{extra} "
                      f"NIQE={m.get('niqe', '-')} BRISQUE={m.get('brisque', '-')}")

            main = next(iter(epoch_scores.values()))
            ref_m = record[next(iter(epoch_scores.keys()))]
            score = main
            if "niqe" in ref_m:
                score -= 0.1 * ref_m["niqe"]
            if "brisque" in ref_m:
                score -= 0.05 * ref_m["brisque"]
            if "laplacian_var" in ref_m:
                score += 1e-3 * ref_m["laplacian_var"]

            if score > best_score:
                best_score = score
                torch.save({"model_state": model.state_dict(),
                            "args": vars(args), "score": best_score}, ckpt_best)

        history.append(record)
        torch.save({"model_state": model.state_dict(),
                    "args": vars(args), "epoch": epoch}, ckpt_last)

    # Final profiling + latency/energy
    sample_lr = torch.randn(1, 3, args.patch_size // args.scale,
                            args.patch_size // args.scale, device=device)
    profile = profile_model(model, sample_lr)
    energy = benchmark_latency(model, sample_lr, n=args.energy_runs)

    # Visualization
    plot_training_curves(history, args.out_dir)
    final_eval: Dict[str, Dict] = {}
    for name, loader in eval_loaders.items():
        final_eval[name] = evaluate(model, loader, args.scale, device,
                                   max_items=args.max_eval)
    plot_metric_bar(final_eval, args.out_dir)
    first_loader = next(iter(eval_loaders.values()))
    save_compare_grid(model, first_loader, args.scale, device,
                      os.path.join(args.out_dir, "compare_grid.png"))

    summary = {
        "args": vars(args), "params": n_params, "s1_params": s1_params,
        "best_score": round(best_score, 4),
        "final_eval": final_eval, "profile": profile, "energy": energy,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "train_results.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[train] done. best score: {best_score:.3f}")
    print(f"[train] profile: {profile}")
    print(f"[train] energy/latency: {energy}")


# ---------------------------------------------------------------------------
# Evaluation / profiling / export
# ---------------------------------------------------------------------------
def _build_model(args, device) -> DualSysBDNetSR:
    return DualSysBDNetSR(
        scale=args.scale, dim=args.dim, blocks=args.blocks,
        alpha=args.alpha, use_bdo=not args.no_bdo, use_s2=not args.no_s2,
        s2_dim=args.s2_dim, s2_bypass_threshold=args.s2_bypass_threshold,
    ).to(device)


@torch.no_grad()
def evaluate_only(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valid_dir = ensure_div2k(args.data_dir)
    model = _build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_loaders = build_eval_loaders(args, valid_dir)
    results: Dict[str, Dict] = {}
    for name, loader in eval_loaders.items():
        results[name] = evaluate(model, loader, args.scale, device,
                                 max_items=args.max_eval, only_s1=args.only_s1)
        print(f"[eval] {name}: {results[name]}")

    sample_lr = torch.randn(1, 3, args.patch_size // args.scale,
                            args.patch_size // args.scale, device=device)
    profile = profile_model(model, sample_lr)
    energy = benchmark_latency(model, sample_lr, n=args.energy_runs)

    plot_metric_bar(results, args.out_dir)
    first_loader = next(iter(eval_loaders.values()))
    save_compare_grid(model, first_loader, args.scale, device,
                      os.path.join(args.out_dir, "compare_grid.png"))

    out = {"ckpt": args.ckpt, "only_s1": args.only_s1, "results": results,
           "profile": profile, "energy": energy}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[eval] profile: {profile}")
    print(f"[eval] energy/latency: {energy}")


def profile_only(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args, device)
    sample_lr = torch.randn(1, 3, args.patch_size // args.scale,
                            args.patch_size // args.scale, device=device)
    profile = profile_model(model, sample_lr)
    energy = benchmark_latency(model, sample_lr, n=args.energy_runs)
    out = {"profile": profile, "energy": energy}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "profile_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def export_onnx(args) -> None:
    device = torch.device("cpu")  # CPU is safer for ONNX export
    model = _build_model(args, device)
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    h = args.patch_size // args.scale
    x = torch.randn(1, 3, h, h)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.only_s1:
        # Pure BDNet(S1) export: single in/out, minimal ops, INT8 quantization friendly
        onnx_path = os.path.join(args.out_dir, "bdnet_s1.onnx")
        torch.onnx.export(
            model.s1, x, onnx_path, opset_version=18,
            input_names=["lr"], output_names=["sr"],
            dynamic_axes={"lr": {0: "batch", 2: "h", 3: "w"},
                          "sr": {0: "batch", 2: "h", 3: "w"}},
        )
    else:
        onnx_path = os.path.join(args.out_dir, "dual_sys_bdnet_sr.onnx")

        class _Wrapper(nn.Module):
            """Keep only the main HR output at export; drop the mask placeholder for easier RKNN parsing."""

            def __init__(self, m: DualSysBDNetSR):
                super().__init__()
                self.m = m

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                final, _, _ = self.m(inp)
                return final

        torch.onnx.export(
            _Wrapper(model), x, onnx_path, opset_version=18,
            input_names=["lr"], output_names=["sr"],
            dynamic_axes={"lr": {0: "batch", 2: "h", 3: "w"},
                          "sr": {0: "batch", 2: "h", 3: "w"}},
        )
    print(f"[export] ONNX -> {onnx_path}")


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DualSys-BDNet-SR embedded-friendly SR pipeline")
    p.add_argument("--mode", choices=["train", "eval", "profile", "export"], default="train")
    p.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])

    # Data
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--train-dir", default="", help="explicit HR training dir (overrides DIV2K auto-download)")
    p.add_argument("--patch-size", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)

    # Model (S1 / ultra-light BDNet)
    p.add_argument("--dim", type=int, default=10, help="backbone channels (8 ~800 params, 10 ~1000 params)")
    p.add_argument("--blocks", type=int, default=1, help="number of residual units (default 1, keeps params low)")
    p.add_argument("--alpha", type=float, default=0.95, help="slow reference EMA momentum")
    p.add_argument("--no-bdo", action="store_true", help="disable BDO low-rank slow-reference modulation")

    # Model (S2 / sparse-gated refinement)
    p.add_argument("--no-s2", action="store_true", help="disable S2 branch, fall back to pure BDNet-SR baseline")
    p.add_argument("--s2-dim", type=int, default=4, help="S2 branch hidden channels")
    p.add_argument("--s1-weight", type=float, default=0.4, help="weight of S1 standalone supervision loss")
    p.add_argument("--mask-target-weight", type=float, default=0.2,
                   help="weight of mask-to-S1-error alignment loss")
    p.add_argument("--sparse-weight", type=float, default=1e-3,
                   help="mask sparsity weight (encourage closed gate on flat regions)")
    p.add_argument("--s2-bypass-threshold", type=float, default=0.0,
                   help="bypass S2 at inference when global mask mean below this; 0 = no bypass")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # IO / Evaluation
    p.add_argument("--ckpt", default="./results/best.pth")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--energy-runs", type=int, default=50)
    p.add_argument("--only-s1", action="store_true",
                   help="use S1 only (BDNet backbone) for eval/export, skip S2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate_only(args)
    elif args.mode == "profile":
        profile_only(args)
    else:
        export_onnx(args)
