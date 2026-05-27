# EVSSM-BAGS: Video Deblurring + Gaussian Splatting Pipeline

Reproduction code for the **blurball** experiment from the DeblurNeRF dataset.  
Pipeline: **EVSSM UnblurSLAM** (per-frame deblurring) → **BAGS (MipSplat)** + **TriSplat ablations** (3D Gaussian scene reconstruction).

## Results (blurball, 600×400, sharp test frames)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | NIMA↑ |
|---|---|---|---|---|
| Input (sharp frames) | — | — | — | 0.7371 |
| EVSSM UnblurSLAM | 35.935 | 0.9882 | 0.0229 | 0.7291 |
| **BAGS (MipSplat)** | **28.705** | **0.8572** | **0.2943** | **0.7243** |
| Abl1: TriSplat + COLMAP | 27.841 | 0.8415 | 0.2845 | 0.6339 |
| Abl2: TriSplat + COLMAP + DA3depth | 23.316 | 0.6701 | 0.4461 | 0.4863 |
| Abl3: TriSplat + DA3pose + DA3depth | 21.547 | 0.6021 | 0.5123 | 0.4674 |

Evaluation: 4 sharp test frames (indices 0, 7, 14, 21 with hold=7), GT = official `images_4/` (600×400).  
All 27 frames used for training (no holdout during training).

---

## Prerequisites

### 1. Clone required repositories

```bash
BASE=/path/to/your/workspace   # set this to your data root

mkdir -p $BASE/repos
cd $BASE/repos

# Triangle-splatting (TriSplat)
git clone https://github.com/shumash/triangle-splatting
# Replace train_bpn.py and arguments/__init__.py with our modified versions:
cp /path/to/this/repo/triangle_splatting/train_bpn.py $BASE/repos/triangle-splatting/
cp /path/to/this/repo/triangle_splatting/arguments__init__.py $BASE/repos/triangle-splatting/arguments/__init__.py

# BAGS (MipSplat)
git clone https://github.com/snldmt/BAGS

# EVSSM
git clone https://github.com/house-of-secrets/EVSSM
```

### 2. Download EVSSM UnblurSLAM checkpoint

Place the checkpoint at:
```
$BASE/checkpoints/evssm_unblurslam/net_g_latest.pth
```

### 3. Set up conda environments

```bash
conda env create -f envs/trigsplat.yml
conda env create -f envs/bags.yml
conda env create -f envs/evssm.yml
```

---

## Dataset

Download the **DeblurNeRF real camera motion blur** dataset:  
[Google Drive — DeblurNeRF dataset](https://drive.google.com/drive/folders/1niCIwMqEGGCc2FqLVyXMmkmZWQpQVJbO)

Place blurball under:
```
$BASE/data/deblurnerf/real_camera_motion_blur_all/blurball/
├── images/          # original 2400×1600 frames (000.jpg … 026.jpg)
├── images_4/        # official 600×400 downsampled GT (000.png … 026.png)
├── sparse/          # COLMAP sparse reconstruction
└── poses_bounds.npy
```

The pipeline also requires DA3-estimated depth and pose directories (`blurball_unblurslam_da3depth/`, `blurball_unblurslam_dav3/`) for Abl2 and Abl3.  
If you only want to reproduce BAGS + Abl1, the base blurball directory is sufficient.

---

## Reproduce

```bash
# Set paths
export BASE=/path/to/your/workspace
export REPO_TRI=$BASE/repos/triangle-splatting
export REPO_BAGS=$BASE/repos/BAGS
export REPO_EVSSM=$BASE/repos/EVSSM
export CKPT_EVSSM=$BASE/checkpoints/evssm_unblurslam/net_g_latest.pth
export GPU1=0
export GPU2=1

bash scripts/run_pipeline.sh
```

The script runs all 5 stages automatically:
1. **EVSSM** deblurring on `images_4/` (600×400)
2. Build scene directories
3. **Train** BAGS + 3 TriSplat ablations (~1.5 hours on 2× A6000)
4. **Render** all training frames
5. **Evaluate** metrics on sharp test frames → printed table + saved to `outputs/logs/metrics_final.log`

---

## Visualization

```bash
# Sharp test frames (0, 7, 14, 21)
conda run -n trigsplat env BASE=$BASE python scripts/visualize_sharp.py

# Blurry training frames (adjacent frames 1, 8, 15, 22)
conda run -n trigsplat env BASE=$BASE python scripts/visualize_blur.py
```

Output saved to `$BASE/outputs/viz_deblurnerf/`.

---

## Repository Structure

```
EVSSM-BAGS/
├── scripts/
│   ├── run_pipeline.sh          # end-to-end pipeline
│   ├── build_scenes.sh          # build scene directories
│   ├── metrics.py               # PSNR/SSIM/LPIPS/NIMA evaluation
│   ├── visualize_sharp.py       # visualization on sharp frames
│   └── visualize_blur.py        # visualization on blurry frames
├── triangle_splatting/
│   ├── train_bpn.py             # modified TriSplat training with BPN
│   └── arguments__init__.py     # modified arguments (max_shapes etc.)
├── envs/
│   ├── trigsplat.yml
│   ├── bags.yml
│   └── evssm.yml
└── README.md
```
