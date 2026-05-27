#!/usr/bin/env bash
# ============================================================
# EVSSM-BAGS: Full blurball 600x400 reproduction pipeline
# Usage:
#   BASE=/path/to/your/data GPU1=0 GPU2=1 bash scripts/run_pipeline.sh
# ============================================================
set -euo pipefail

BASE="${BASE:-/data/blur_slam_exp}"
GPU1="${GPU1:-0}"
GPU2="${GPU2:-1}"

DATA="$BASE/data/deblurnerf/real_camera_motion_blur_all"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_TRI="${REPO_TRI:-$BASE/repos/triangle-splatting}"
REPO_BAGS="${REPO_BAGS:-$BASE/repos/BAGS}"
REPO_EVSSM="${REPO_EVSSM:-$BASE/repos/EVSSM}"
CKPT_EVSSM="${CKPT_EVSSM:-$BASE/checkpoints/evssm_unblurslam/net_g_latest.pth}"
LOG="$BASE/outputs/logs"
mkdir -p "$LOG"

EVSSM_OUT="$BASE/data/evssm_deblurred_deblurnerf/blurball_unblurslam_600"
SCENE_BASE="$DATA/blurball_evssm600"
OUT_TRI="$BASE/outputs/trigsplat_deblurnerf_600v2"
OUT_BAGS="$BASE/outputs/bags_deblurnerf_600v2"

echo "=================================================="
echo "EVSSM-BAGS Pipeline  [$(date)]"
echo "  BASE     = $BASE"
echo "  GPU1=$GPU1  GPU2=$GPU2"
echo "=================================================="

# ── Stage 1: EVSSM deblurring on 600×400 frames ─────────────────────────────
echo "[1/5] EVSSM UnblurSLAM deblurring ..."
STAGING="/tmp/evssm_blurball600_input/blurball"
mkdir -p "$STAGING/test/input" "$STAGING/test/target"
for F in "$DATA/blurball/images_4/"*.png; do
    FNAME=$(basename "$F")
    ln -sf "$F" "$STAGING/test/input/$FNAME"
    ln -sf "$F" "$STAGING/test/target/$FNAME"
done

mkdir -p "$EVSSM_OUT"
conda run -n evssm --cwd "$REPO_EVSSM" \
    env CUDA_VISIBLE_DEVICES=$GPU1 \
    python test.py \
        --data_dir "$STAGING" \
        --test_model "$CKPT_EVSSM" \
        --model_name blurball_600 \
    > "$LOG/evssm_blurball600.log" 2>&1

# Collect output (EVSSM saves to results_final_2/blurball_600/GoPro/)
IDX=0
for F in $(find "$REPO_EVSSM/results_final_2/blurball_600/GoPro" \
           -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) | sort); do
    cp "$F" "$(printf '%s/%03d.png' "$EVSSM_OUT" "$IDX")"
    IDX=$((IDX+1))
done
echo "  EVSSM done: $IDX frames → $EVSSM_OUT"

# ── Stage 2: Build scene directories ────────────────────────────────────────
echo "[2/5] Building scene directories ..."
bash "$REPO_ROOT/scripts/build_scenes.sh" "$BASE" "$DATA" "$EVSSM_OUT" "$SCENE_BASE"

# ── Stage 3: Train all models (BAGS + Abl1 on GPU1, Abl2 + Abl3 on GPU2) ───
echo "[3/5] Training ..."

run_trigsplat() {
    local GPU=$1 DATA_DIR=$2 OUT=$3 TAG=$4 EXTRA="${5:-}"
    mkdir -p "$OUT"
    echo "  [train] $TAG gpu=$GPU $(date +%H:%M:%S)"
    conda run -n trigsplat --cwd "$REPO_TRI" \
        env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 \
        python train_bpn.py \
            -s "$DATA_DIR" -m "$OUT" \
            --images images_4 -r 1 \
            --iterations 15000 --test_iterations 15000 --save_iterations 15000 \
            --densify_from_iter 200 --densify_until_iter 7000 \
            --max_shapes 800000 \
            --train_bpn \
            --kernel_size1 5 --kernel_size2 9 --kernel_size3 21 --kernel_size_ss 21 \
            --mask_loss_alpha 0.001 \
            $EXTRA --quiet
    echo "  [done]  $TAG $(date +%H:%M:%S)"
}

run_bags() {
    local GPU=$1 DATA_DIR=$2 OUT=$3
    mkdir -p "$OUT"
    echo "  [train-bags] gpu=$GPU $(date +%H:%M:%S)"
    conda run -n bags --cwd "$REPO_BAGS" \
        env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 \
        python train.py \
            -s "$DATA_DIR" -m "$OUT" \
            --images images_4 -r 1 \
            --iterations 15000 --ms_steps 6000 \
            --test_iterations 15000 --save_iterations 15000 \
            --quiet
    echo "  [done-bags] $(date +%H:%M:%S)"
}

(
    run_bags $GPU1 "${SCENE_BASE}_nodepth" "$OUT_BAGS/blurball_evssm600_nodepth" \
        >> "$LOG/bags_blurball600.log" 2>&1
    run_trigsplat $GPU1 "${SCENE_BASE}_nodepth" \
        "$OUT_TRI/blurball_evssm600_nodepth" "Abl1" \
        >> "$LOG/abl1.log" 2>&1
) &
PID1=$!

(
    run_trigsplat $GPU2 "${SCENE_BASE}_da3depth" \
        "$OUT_TRI/blurball_evssm600_da3depth" "Abl2" \
        "--use_depth_loss --depth_loss_alpha 0.1 --depth_scale_invariant" \
        >> "$LOG/abl2.log" 2>&1
    run_trigsplat $GPU2 "${SCENE_BASE}_dav3" \
        "$OUT_TRI/blurball_evssm600_dav3" "Abl3" \
        "--use_depth_loss --depth_loss_alpha 0.1 --depth_scale_invariant" \
        >> "$LOG/abl3.log" 2>&1
) &
PID2=$!

wait $PID1 $PID2
echo "  Training done: $(date)"

# ── Stage 4: Render ──────────────────────────────────────────────────────────
echo "[4/5] Rendering ..."
conda run -n bags --cwd "$REPO_BAGS" env CUDA_VISIBLE_DEVICES=$GPU1 \
    python render.py -m "$OUT_BAGS/blurball_evssm600_nodepth" --iteration 15000 \
    --skip_test --quiet >> "$LOG/render_bags.log" 2>&1

for abl in nodepth da3depth dav3; do
    conda run -n trigsplat --cwd "$REPO_TRI" env CUDA_VISIBLE_DEVICES=$GPU1 \
        python render.py -m "$OUT_TRI/blurball_evssm600_${abl}" --iteration 15000 \
        --quiet >> "$LOG/render_${abl}.log" 2>&1
done
echo "  Render done"

# ── Stage 5: Metrics ─────────────────────────────────────────────────────────
echo "[5/5] Computing metrics ..."
conda run -n trigsplat \
    env BASE="$BASE" \
    python "$REPO_ROOT/scripts/metrics.py" | tee "$LOG/metrics_final.log"

echo "=================================================="
echo "DONE  [$(date)]"
echo "Results saved to: $LOG/metrics_final.log"
echo "=================================================="
