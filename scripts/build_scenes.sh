#!/usr/bin/env bash
# Build scene directories for BAGS and TriSplat ablations
# Usage: bash build_scenes.sh BASE DATA EVSSM_OUT SCENE_BASE
set -euo pipefail

BASE="$1"
DATA="$2"
EVSSM_OUT="$3"
SCENE_BASE="$4"

copy_sparse() {
    local SRC="$1" DST="$2"
    mkdir -p "$DST/sparse/0"
    for f in cameras.bin images.bin points3D.ply; do
        [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/sparse/0/"
    done
    [ -f "$SRC/points3D.bin" ] && cp "$SRC/points3D.bin" "$DST/sparse/0/" || true
}

symlink_evssm() {
    local DST="$1"
    mkdir -p "$DST/images_4"
    for F in "$EVSSM_OUT/"*.png; do
        FNAME=$(basename "$F")
        ln -sf "$F" "$DST/images_4/$FNAME"
    done
}

# Abl1 + BAGS: COLMAP poses, no depth
symlink_evssm "${SCENE_BASE}_nodepth"
copy_sparse   "$DATA/blurball/sparse/0" "${SCENE_BASE}_nodepth"

# Abl2: COLMAP poses + DA3 depth
symlink_evssm "${SCENE_BASE}_da3depth"
copy_sparse   "$DATA/blurball_unblurslam_da3depth/sparse/0" "${SCENE_BASE}_da3depth"
ln -sfn "$DATA/blurball_unblurslam_da3depth/depth" "${SCENE_BASE}_da3depth/depth" 2>/dev/null || true

# Abl3: DA3 poses + DA3 depth
symlink_evssm "${SCENE_BASE}_dav3"
copy_sparse   "$DATA/blurball_unblurslam_dav3/sparse/0" "${SCENE_BASE}_dav3"
ln -sfn "$DATA/blurball_unblurslam_dav3/depth" "${SCENE_BASE}_dav3/depth" 2>/dev/null || true

echo "  Scene dirs built:"
echo "    ${SCENE_BASE}_nodepth"
echo "    ${SCENE_BASE}_da3depth"
echo "    ${SCENE_BASE}_dav3"
