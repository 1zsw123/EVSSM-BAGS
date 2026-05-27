#!/usr/bin/env python3
"""Zoomed crop visualization for blurball 600×400 — 2 selected frames, center crop."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE     = "/home/szha0669/storage/blur_slam_exp"
DATA     = f"{BASE}/data/deblurnerf/real_camera_motion_blur_all"
OUT_TRI  = f"{BASE}/outputs/trigsplat_deblurnerf_600"
OUT_BAGS = f"{BASE}/outputs/bags_deblurnerf_600"

N_TOTAL = 27; HOLD = 7
test_ids = [i for i in range(N_TOTAL) if i % HOLD == 0]  # [0, 7, 14, 21]

METHODS = [
    ("Input (blurry)",         None, "blur"),
    ("GT (images_4)",          None, "gt"),
    ("EVSSM UnblurSLAM",       None, "evssm"),
    ("BAGS MipSplat",          f"{OUT_BAGS}/blurball_evssm600_nodepth/test/ours_15000/test_preds_1", "render"),
    ("Abl1 TriSplat+COLMAP",   f"{OUT_TRI}/blurball_evssm600_nodepth/test/ours_15000/test_preds_1",  "render"),
    ("Abl2 +DA3depth",         f"{OUT_TRI}/blurball_evssm600_da3depth/test/ours_15000/test_preds_1", "render"),
    ("Abl3 DA3pose+depth",     f"{OUT_TRI}/blurball_evssm600_dav3/test/ours_15000/test_preds_1",     "render"),
]

# Collect render files per method
render_files = {}
for ci, (label, rdir, kind) in enumerate(METHODS):
    if rdir and os.path.isdir(rdir):
        render_files[ci] = sorted([f for f in os.listdir(rdir) if f.endswith('.png')])

def load_frame(ci, ri, frame_idx):
    label, rdir, kind = METHODS[ci]
    if kind == "blur":
        for ext in ['.jpg', '.png']:
            p = os.path.join(DATA, "blurball/images", f"{frame_idx:03d}{ext}")
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
    elif kind == "gt":
        p = os.path.join(DATA, "blurball/images_4", f"{frame_idx:03d}.png")
        if os.path.exists(p):
            return Image.open(p).convert('RGB')
    elif kind == "evssm":
        p = os.path.join(BASE, "data/evssm_deblurred_deblurnerf/blurball_unblurslam_600",
                         f"{frame_idx:03d}.png")
        if os.path.exists(p):
            return Image.open(p).convert('RGB')
    elif kind == "render" and ci in render_files:
        files = render_files[ci]
        if ri < len(files):
            p = os.path.join(rdir, files[ri])
            return Image.open(p).convert('RGB')
    return None

# Layout: use all 4 rows, all 7 columns, full 600×400 images
CELL_W, CELL_H = 600, 400
LABEL_H = 32
PAD = 3
N_COLS = len(METHODS)
N_ROWS = len(test_ids)

canvas_w = PAD + N_COLS * (CELL_W + PAD)
canvas_h = LABEL_H + PAD + N_ROWS * (CELL_H + PAD)

canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
draw = ImageDraw.Draw(canvas)

try:
    font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font_hdr = ImageFont.load_default()
    font_lbl = font_hdr

# Column headers
for ci, (label, _, _) in enumerate(METHODS):
    x = PAD + ci * (CELL_W + PAD)
    draw.rectangle([x, 0, x + CELL_W, LABEL_H - 1], fill=(50, 50, 70))
    draw.text((x + CELL_W // 2, LABEL_H // 2), label,
              font=font_hdr, fill=(240, 240, 255), anchor="mm")

# Images
for ri, frame_idx in enumerate(test_ids):
    y = LABEL_H + PAD + ri * (CELL_H + PAD)
    for ci in range(N_COLS):
        x = PAD + ci * (CELL_W + PAD)
        img = load_frame(ci, ri, frame_idx)
        if img is not None:
            img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
            canvas.paste(img, (x, y))
            # Frame label on first column
            if ci == 0:
                draw.text((x + 5, y + 5), f"Frame {frame_idx:03d}",
                          font=font_lbl, fill=(255, 255, 80))
        else:
            draw.rectangle([x, y, x + CELL_W, y + CELL_H], fill=(80, 30, 30))
            draw.text((x + CELL_W//2, y + CELL_H//2), "N/A",
                      font=font_hdr, fill=(200, 80, 80), anchor="mm")

out = f"{BASE}/outputs/viz_deblurnerf/blurball600_full.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
canvas.save(out)
print(f"Saved: {out}  size={canvas.size}")
