# DualSys-BDNet-SR：千参数级双系统轻量化图像超分

> 面向边缘端 / 低功耗设备（RK3588 等）的轻量超分网络。**S1 = BDNet 快通路直觉重建，S2 = 稀疏门控选择性审慎修正**；全程标准算子，无注意力、无 BN、无 KV Cache、无 VLM 依赖，可一键导出 ONNX 供 RKNN / TFLite 量化部署。

- **模型规模（x4，默认配置实测）**：参数量 **1018**（S1≈970，S2≈48），MACs ≈ **0.25 M**。
- **单文件流水线**：数据加载 → 训练 → 有/无参考指标评估 → 参数量/MACs/流量/能耗/延迟 profiling → 可视化 → ONNX 导出，全部在 `dual_sys_bdnet_sr.py`。

---

## 1. 设计动机：双系统与人脑视觉分工

借鉴 Kahneman 双系统理论，并对应人脑视觉的“先粗看全局、再对可疑区域二次细看”：

| 通路 | 对应人脑 | 职责 | 实现 |
|------|---------|------|------|
| **System1（快通路）** | 腹侧 feedforward 直觉感知 | 一眼完成全局粗重建，低算力、可独立剥离部署 | 超轻量 BDNet 主干：窄通道 + 单残差单元，深度可分离 3×3 + 1×1 通道混合，1×1 子像素上采样；保留 BDO / EMA 低秩慢参考调制 |
| **System2（慢通路）** | 审慎、选择性再检验 | 只在纹理/边缘/误差大的区域开门做局部残差校正，平坦区域关门、几乎不引入额外计算 | 全 1×1 卷积稀疏门控分支：输出修正项 `delta` 与空间门控 `mask`，`refine = delta * mask` |

与传统全局等算力堆叠残差/注意力不同，S2 是**条件式、稀疏**的：训练期用 S1 与 GT 的逐像素误差监督门控，使模型学会“哪里需要再检验”；推理期可按 mask 全局均值自适应旁路 S2，平坦帧直接只跑 S1，进一步压低功耗。

## 2. 模型结构

- **S1 / MicroBDNetSR**
  - 输入归一化 → 3×3 卷积提特征 → 单个深度可分离残差单元 → 全局 LR 残差连接；
  - BDO（低秩 rank=1，EMA 慢参考通道调制，训练期更新参考、推理期冻结）；
  - 1×1 子像素卷积直接预测 `3·r²` 通道，PixelShuffle 单次上采样。
- **S2 / System2Refine**
  - 拼接 S1 输出与 bicubic 参考上采样 → 1×1 隐藏层 → 分别预测 `delta` 与 sigmoid 门控 `mask`。
- **算子清单**：`Conv2d(3×3 / 1×1, 含深度可分离) / PixelShuffle / ReLU6 / Sigmoid`，无 permute 缓冲、无自定义算子、无注意力。

### 规模可调（x4）

| `--dim` | `--blocks` | 总参数量 |
|---------|-----------|---------|
| 6  | 1 | ≈606 |
| 8  | 1 | ≈808 |
| 10 | 1 | ≈1018（默认） |

## 3. 快速开始

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 数据准备

- 不手动准备时，脚本自动下载 DIV2K train/valid 到 `./data/DIV2K/`。
- 也可用 `--train-dir` 显式指定自己的 HR 图片目录；LR 由 HR 训练时实时 bicubic 下采样生成，无需单独准备 LR。
- 标准 benchmark（Set5 / Set14 / B100 / Urban100）放到 `./data/benchmark/<Name>/*.png`，评估时自动识别；缺失则回退 DIV2K valid。

### 3.3 训练

```bash
# 完整双系统（默认千参配置，x4）
python dual_sys_bdnet_sr.py --mode train --scale 4 --dim 10 --blocks 1 --epochs 100

# 纯 BDNet 基线（关闭 S2）
python dual_sys_bdnet_sr.py --mode train --scale 4 --no-s2

# 进一步压参：dim=8 + 关闭 BDO
python dual_sys_bdnet_sr.py --mode train --scale 4 --dim 8 --no-bdo
```

### 3.4 评估

```bash
# 完整双系统
python dual_sys_bdnet_sr.py --mode eval --scale 4 --ckpt ./results/best.pth

# 只评估 S1（BDNet 本体，跳过 S2）
python dual_sys_bdnet_sr.py --mode eval --scale 4 --ckpt ./results/best.pth --only-s1
```

评估指标：
- **有参考（Y 通道，BT.601，裁边界）**：PSNR-Y、SSIM-Y（同时记录 S1 单独结果用于对照）。
- **无参考**：Laplacian 方差、梯度能量；安装 `piq` 后补充 NIQE / BRISQUE / PIQE。
- **效率/能耗**：参数量、MACs、激活 DRAM 流量、启发式流量能耗、逐图延迟；有 NVIDIA GPU + pynvml 时记录平均功耗与单图能量。

### 3.5 Profiling 与导出

```bash
# 只看参数量 / MACs / 流量 / 延迟 / 能耗
python dual_sys_bdnet_sr.py --mode profile --scale 4 --dim 10 --blocks 1

# 导出完整双系统 ONNX
python dual_sys_bdnet_sr.py --mode export --scale 4 --ckpt ./results/best.pth

# 导出纯 BDNet(S1) ONNX，算子最精简，适合 INT8 量化
python dual_sys_bdnet_sr.py --mode export --scale 4 --ckpt ./results/best.pth --only-s1
```

## 4. 结果与对比

> 模型复杂度为仓库默认配置实测；画质指标需在 DIV2K / benchmark 上完成训练后填入（见 `comparison_table.md`），未训练前不预设数值。

| Model | Params | MACs | PSNR-Y | SSIM-Y | 延迟 | 备注 |
|-------|--------|------|--------|--------|------|------|
| **DualSys-BDNet-SR (Ours)** | 1018 | ≈0.25 M | _待训练填入_ | _待训练填入_ | _待实测_ | 双系统稀疏修正，S1 可独立剥离 |
| S1-BDNet（单独） | 970 | ≈0.22 M | _待训练填入_ | _待训练填入_ | _待实测_ | 基线，无 S2 修正分支 |

公开轻量方法的参数 / MACs 参考值见 `comparison_table.md`，对比前需在同一数据集与协议下复现核对。

## 5. 输出产物（`--out-dir`，默认 `./results`）

| 文件 | 内容 |
|------|------|
| `best.pth` / `last.pth` | 最优 / 最近一轮 checkpoint |
| `train_results.json` / `eval_results.json` | 全部指标、profiling、能耗、训练历史 |
| `compare_grid.png` | LR / Bicubic / SR(Ours) / S1(BDNet) / HR / S2 Mask 对比网格 |
| `training_curves.png` | 训练 L1、PSNR-Y、SSIM-Y 曲线 |
| `s2_mask_curve.png` | S2 门控活跃度随训练曲线 |
| `metric_bar.png` | 各 benchmark 指标柱状图 |

## 6. 嵌入式落地说明

- 全程 LR 空间运算，末尾单次 PixelShuffle；深度可分离卷积 + ReLU6。
- 标准 `Conv2d / PixelShuffle / ReLU6 / Sigmoid`，无 permute、无自定义算子、无注意力 / KV Cache。
- ONNX 可经 RKNN Toolkit 转 RK3588 等平台；如需 8bit 部署，可再加量化感知训练（QAT）。
- 视频流场景可按 `--s2-bypass-threshold` 动态旁路 S2，退化到纯 S1-BDNet 进一步降能耗。

## 7. 仓库结构

```
DualSys-BDNet-SR/
├── dual_sys_bdnet_sr.py   # 主流水线：模型 / 数据 / 训练 / 评估 / profiling / 可视化 / 导出
├── comparison_table.md     # 轻量超分方法对比表（参数 / MACs / 画质 / 能耗）
├── requirements.txt       # 依赖清单
├── LICENSE                # MIT License
└── .gitignore
```

## 8. 关键命令行参数

| 参数 | 作用 | 默认 |
|------|------|------|
| `--dim` | 主干通道数（8≈800 参，10≈1000 参） | 10 |
| `--blocks` | 残差单元数 | 1 |
| `--alpha` | BDO 慢参考 EMA 动量 | 0.95 |
| `--no-bdo` | 关闭 BDO 低秩调制 | 关 |
| `--no-s2` | 关闭 S2 稀疏修正分支 | 关 |
| `--s2-dim` | S2 隐藏通道数 | 4 |
| `--s1-weight` | S1 独立监督 loss 权重 | 0.4 |
| `--mask-target-weight` | 门控对齐 S1 误差 loss 权重 | 0.2 |
| `--sparse-weight` | 门控稀疏正则权重 | 1e-3 |
| `--s2-bypass-threshold` | 推理期 mask 均值低于阈值时旁路 S2（0=不旁路） | 0.0 |
| `--epochs` / `--batch-size` / `--lr` | 训练超参 | 100 / 16 / 1.5e-3 |

## 9. 复现与局限说明

- 千参数级模型属于**极限轻量**设定，纹理恢复能力受限，适合算力/功耗敏感、对峰值 PSNR 要求不极端的端侧场景；若细节不足，优先提高 `--dim`（12–16，约 1.5k–2.5k 参），比堆叠残差块更划算。
- 仓库内画质指标占位需在标准 benchmark 上训练后实测填写；对比表中他方方法数值为公开文献参考，需统一协议复现后再横向比较。

## Citation

```bibtex
@misc{zheng2026dualsysbdnet,
  title={DualSys-BDNet-SR: A 1k-Parameter Dual-System Lightweight Super-Resolution Network for Edge Low-Power Deployment},
  author={Zhuoran Zheng},
  year={2026},
  howpublished={GitHub repository}
}
```
