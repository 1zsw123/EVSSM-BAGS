#!/usr/bin/env python3
"""
Metrics for blurball 600×400 v2 (all 27 frames training, no holdout).
GT = images_4/ (600×400). Evaluate frames 0,7,14,21 from train renders.
Render output: train/ours_15000/test_preds_1/00000.png … 00026.png
Sharp test frames map: idx 0→file 00000, idx 7→file 00007, idx 14→file 00014, idx 21→file 00021
"""
import os, json
import numpy as np
from PIL import Image
import torch, pyiqa, lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

BASE = os.environ.get("BASE", "/home/szha0669/storage/blur_slam_exp")
GT_DIR   = f"{BASE}/data/deblurnerf/real_camera_motion_blur_all/blurball/images_4"
BLUR_DIR = f"{BASE}/data/deblurnerf/real_camera_motion_blur_all/blurball/images"
EVSSM600 = f"{BASE}/data/evssm_deblurred_deblurnerf/blurball_unblurslam_600"
TRI_OUT  = f"{BASE}/outputs/trigsplat_deblurnerf_600v2"
BAGS_OUT = f"{BASE}/outputs/bags_deblurnerf_600v2"

N_TOTAL = 27; HOLD = 7
test_ids = [i for i in range(N_TOTAL) if i % HOLD == 0]  # [0, 7, 14, 21]
GT_SIZE  = (600, 400)

device   = torch.device("cuda")
nima_fn  = pyiqa.create_metric("nima-koniq", device=device)
lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)

def t(img):
    return (torch.from_numpy(np.array(img).astype(np.float32) / 255.)
            .permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1)

def eval_pred_vs_gt(pred_img, gt_img):
    if pred_img.size != GT_SIZE:
        pred_img = pred_img.resize(GT_SIZE, Image.LANCZOS)
    p = np.array(pred_img); g = np.array(gt_img)
    psnr = psnr_fn(g, p, data_range=255)
    ssim = ssim_fn(g, p, channel_axis=2, data_range=255)
    with torch.no_grad():
        lp = float(lpips_fn(t(pred_img), t(gt_img)).item())
    return dict(psnr=psnr, ssim=ssim, lpips=lp)

gt_imgs = [Image.open(os.path.join(GT_DIR, f"{i:03d}.png")).convert('RGB') for i in test_ids]

print(f"GT: {GT_DIR}")
print(f"Sharp test frames: {test_ids}  (N={len(test_ids)})")
print()

results = {}

def eval_files(pred_paths, label):
    rows = []
    for pred_path, gt_img in zip(pred_paths, gt_imgs):
        pred_img = Image.open(pred_path).convert('RGB')
        m = eval_pred_vs_gt(pred_img, gt_img)
        m['nima'] = float(nima_fn(pred_path))
        rows.append(m)
    return rows

def eval_train_renders(render_dir, label):
    """Pick frames 0,7,14,21 from train renders (named 00000.png…00026.png)."""
    rows = []
    for fi, gt_img in zip(test_ids, gt_imgs):
        pred_path = os.path.join(render_dir, f"{fi:05d}.png")
        if not os.path.exists(pred_path):
            print(f"  WARNING: missing {pred_path}")
            continue
        pred_img = Image.open(pred_path).convert('RGB')
        m = eval_pred_vs_gt(pred_img, gt_img)
        m['nima'] = float(nima_fn(pred_path))
        rows.append(m)
    return rows

def summarise(rows, label):
    if not rows:
        print(f"  {label}: NO DATA"); return None
    r = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    r['n'] = len(rows)
    results[label] = r
    return r

def print_row(label, r, indent=2):
    if r is None: return
    print(f"{' '*indent}{label:<48} N={r['n']}  PSNR={r['psnr']:.3f}  SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}  NIMA={r['nima']:.4f}")

# 1. Input blurry
print("Computing: Input (blurry) ...")
blur_paths = []
for i in test_ids:
    for ext in ['.jpg', '.png']:
        p = os.path.join(BLUR_DIR, f"{i:03d}{ext}")
        if os.path.exists(p): blur_paths.append(p); break
r = summarise(eval_files(blur_paths, "Input (blurry)"), "Input (blurry)")
print_row("Input (blurry)", r)

# 2. EVSSM 600
print("Computing: EVSSM UnblurSLAM 600 ...")
evssm_paths = [os.path.join(EVSSM600, f"{i:03d}.png") for i in test_ids]
if all(os.path.exists(p) for p in evssm_paths):
    r = summarise(eval_files(evssm_paths, "EVSSM UnblurSLAM 600"), "EVSSM UnblurSLAM 600")
    print_row("EVSSM UnblurSLAM 600", r)

# 3. BAGS
print("Computing: BAGS MipSplat ...")
d = f"{BAGS_OUT}/blurball_evssm600_nodepth/train/ours_15000/test_preds_1"
if os.path.isdir(d):
    r = summarise(eval_train_renders(d, "BAGS MipSplat"), "BAGS MipSplat")
    print_row("BAGS MipSplat", r)
else:
    print(f"  BAGS: not found: {d}")

# 4. Abl1
print("Computing: Abl1 no depth ...")
d = f"{TRI_OUT}/blurball_evssm600_nodepth/train/ours_15000/test_preds_1"
if os.path.isdir(d):
    r = summarise(eval_train_renders(d, "Abl1 no depth"), "Abl1 no depth")
    print_row("Abl1 no depth", r)
else:
    print(f"  Abl1: not found: {d}")

# 5. Abl2
print("Computing: Abl2 DA3depth ...")
d = f"{TRI_OUT}/blurball_evssm600_da3depth/train/ours_15000/test_preds_1"
if os.path.isdir(d):
    r = summarise(eval_train_renders(d, "Abl2 DA3depth"), "Abl2 DA3depth")
    print_row("Abl2 DA3depth", r)
else:
    print(f"  Abl2: not found: {d}")

# 6. Abl3
print("Computing: Abl3 DA3pose+depth ...")
d = f"{TRI_OUT}/blurball_evssm600_dav3/train/ours_15000/test_preds_1"
if os.path.isdir(d):
    r = summarise(eval_train_renders(d, "Abl3 DA3pose+depth"), "Abl3 DA3pose+depth")
    print_row("Abl3 DA3pose+depth", r)
else:
    print(f"  Abl3: not found: {d}")

print()
print("=" * 90)
print(f"BLURBALL 600×400 v2 — Sharp frames (N={len(test_ids)}, all 27 used for training)")
print("=" * 90)
print(f"{'Method':<50} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'NIMA':>7}")
print("-" * 78)
for label, r in results.items():
    psnr = f"{r['psnr']:.3f}" if 'psnr' in r else "—"
    ssim = f"{r['ssim']:.4f}" if 'ssim' in r else "—"
    lp   = f"{r['lpips']:.4f}" if 'lpips' in r else "—"
    nm   = f"{r['nima']:.4f}"
    print(f"{label:<50} {psnr:>7} {ssim:>7} {lp:>7} {nm:>7}")

out = f"{BASE}/outputs/logs/metrics_blurball600v2.json"
json.dump(results, open(out, 'w'), indent=2)
print(f"\nSaved: {out}")
