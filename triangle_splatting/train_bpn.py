#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F  # noqa: F401 (used in depth loss)
from random import randint
from utils.loss_utils import l1_loss, ssim, equilateral_regularizer, l2_loss
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state

# ── BPN imports: same BlurKernel modules as BAGS, vendored into this repo ───
from scene.kernelnet_multires import BlurKernel as BlurKernel_ms
from scene.kernelnet_single_res import BlurKernel as BlurKernel_ss
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips


BAGS_CAPTURE_MLP_SS_IDX = 11
BAGS_CAPTURE_MLP_MS_IDX = 12


def get_2d_emb(batch_size, x, y, out_ch, device):
    """BAGS positional encoding used by the blur-kernel predictor."""
    out_ch = int(np.ceil(out_ch / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2, device=device).float() / out_ch))
    pos_x = torch.arange(x, device=device).type(inv_freq.type()) * 2 * np.pi / x
    pos_y = torch.arange(y, device=device).type(inv_freq.type()) * 2 * np.pi / y

    def emb(sin_inp):
        return torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1).flatten(-2, -1)

    emb_x = emb(torch.einsum("i,j->ij", pos_x, inv_freq)).unsqueeze(1)
    emb_y = emb(torch.einsum("i,j->ij", pos_y, inv_freq))
    out = torch.zeros((x, y, out_ch * 2), device=device)
    out[:, :, :out_ch] = emb_x
    out[:, :, out_ch:] = emb_y
    return out[None].repeat(batch_size, 1, 1, 1)


def load_nima_weights(nima_path, source_path):
    """Mirror BAGS' ScanNet frame-id remapping for per-frame NIMA weights."""
    weights = {}
    if not nima_path or not os.path.exists(nima_path):
        return weights

    scene_name = os.path.basename(source_path)
    with open(nima_path) as f:
        all_nima = json.load(f)
    if scene_name not in all_nima:
        return weights

    raw = all_nima[scene_name]
    vals = list(raw.values())
    mean_score = float(np.mean(vals)) if vals else 1.0

    manifest_path = os.path.join(source_path, "conversion_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            frame_ids = json.load(f).get("frame_ids", [])
        for evssm_idx, scannet_id in enumerate(frame_ids):
            key = f"{evssm_idx:06d}.png"
            if key in raw:
                weights[f"{scannet_id:06d}"] = float(np.clip(raw[key] / mean_score, 0.5, 2.0))
    else:
        weights = {
            k.split(".")[0]: float(np.clip(v / mean_score, 0.5, 2.0))
            for k, v in raw.items()
        }

    if weights:
        print(f"[NIMA] loaded {len(weights)} weights, mean={mean_score:.3f}, "
              f"range=[{min(weights.values()):.3f}, {max(weights.values()):.3f}]")
    return weights


def load_gt_depth_cache(scene, source_path):
    """Load ScanNet depth PNGs in meters for GT depth supervision."""
    cache = {}
    depth_dir = os.path.join(source_path, "depth")
    if not os.path.isdir(depth_dir):
        return cache

    for cam in scene.getTrainCameras():
        dpath = os.path.join(depth_dir, cam.image_name + ".png")
        if not os.path.exists(dpath):
            continue
        raw = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        depth_m = raw.astype(np.float32) / 1000.0
        cache[cam.image_name] = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)  # keep on CPU

    print(f"[depth] cached {len(cache)}/{len(scene.getTrainCameras())} GT depth maps")
    return cache


def build_bpn_modules(n_cams, h, w, ks1, ks2, ks3, ks_ss, args):
    not_use_rgbd = getattr(args, "not_use_rgbd", False)
    not_use_pe = getattr(args, "not_use_pe", False)
    mlp_ms = BlurKernel_ms(
        n_cams, h, w, ks1=ks1, ks2=ks2, ks3=ks3,
        not_use_rgbd=not_use_rgbd, not_use_pe=not_use_pe,
    ).cuda()
    mlp_ss = BlurKernel_ss(
        n_cams, h, w, ks=ks_ss,
        not_use_rgbd=not_use_rgbd, not_use_pe=not_use_pe,
    ).cuda()
    return mlp_ms, mlp_ss


def load_bags_bpn_checkpoint(bpn_ckpt, mlp_ms, mlp_ss, scene_ncams, h, w, ks1, ks2, ks3, ks_ss, args):
    if not bpn_ckpt:
        print("[BPN] no BAGS checkpoint provided, training BPN from scratch")
        return mlp_ms, mlp_ss
    if not os.path.exists(bpn_ckpt):
        raise FileNotFoundError(f"BPN checkpoint not found: {bpn_ckpt}")

    bags_data = torch.load(bpn_ckpt, map_location="cpu", weights_only=False)
    model_args = bags_data[0] if isinstance(bags_data, (tuple, list)) else bags_data
    if not isinstance(model_args, (tuple, list)) or len(model_args) <= BAGS_CAPTURE_MLP_MS_IDX:
        raise ValueError(f"Unsupported BAGS checkpoint capture format in {bpn_ckpt}")

    ss_dict = model_args[BAGS_CAPTURE_MLP_SS_IDX]
    ms_dict = model_args[BAGS_CAPTURE_MLP_MS_IDX]
    required_ss = {"embedding_camera.weight", "mlp_base_mlp.0.weight", "mlp_head1.weight", "mlp_mask1.weight"}
    required_ms = {"embedding_camera.weight", "mlp_base1.0.weight", "mlp_head1.weight", "mlp_mask1.weight"}
    if not required_ss.issubset(ss_dict.keys()):
        raise ValueError(f"BAGS checkpoint index {BAGS_CAPTURE_MLP_SS_IDX} does not look like mlp_rgb_ss")
    if not required_ms.issubset(ms_dict.keys()):
        raise ValueError(f"BAGS checkpoint index {BAGS_CAPTURE_MLP_MS_IDX} does not look like mlp_rgb_ms")

    ckpt_ncams = int(ms_dict["embedding_camera.weight"].shape[0])
    if ckpt_ncams != scene_ncams:
        print(f"[BPN] warning: camera embedding count differs (scene={scene_ncams}, ckpt={ckpt_ncams}). "
              "Using checkpoint size; verify llffhold/source match.")
        mlp_ms, mlp_ss = build_bpn_modules(ckpt_ncams, h, w, ks1, ks2, ks3, ks_ss, args)

    mlp_ms.load_state_dict(ms_dict, strict=True)
    mlp_ss.load_state_dict(ss_dict, strict=True)
    print(f"[BPN] loaded BAGS blur kernels from {bpn_ckpt}")
    return mlp_ms, mlp_ss


def apply_bpn_blur(image, depth, cam_idx, iteration, bpn):
    """Apply BAGS' per-pixel blur kernel to a rendered Triangle image."""
    if iteration <= 250:
        return image, torch.zeros_like(image[:1]), None

    shuffle_rgb = image.unsqueeze(0)
    shuffle_depth = depth.unsqueeze(0)
    shuffle_depth = (shuffle_depth - shuffle_depth.min()) / (shuffle_depth.max() - shuffle_depth.min() + 1e-8)
    pos_enc = get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, image.device)
    bpn_input = torch.cat([shuffle_rgb, shuffle_depth], 1).detach()

    if bpn.get("no_curriculum", False):
        # Direct large kernel: skip progressive curriculum, always use ks3 head
        bpn_step = 9999
        if bpn["ks_ss"] != bpn["ks3"]:
            kw, mask = bpn["mlp_ss"](cam_idx, pos_enc, bpn_input, bpn_step)
            kernel_size = bpn["ks_ss"]
        else:
            kw, mask = bpn["mlp_ms"](cam_idx, pos_enc, bpn_input, bpn_step)
            kernel_size = bpn["ks3"]
    elif iteration < 3000:
        kw, mask = bpn["mlp_ms"](cam_idx, pos_enc, bpn_input, iteration)
        kernel_size = bpn["ks1"]
    elif iteration < 6000:
        kw, mask = bpn["mlp_ms"](cam_idx, pos_enc, bpn_input, iteration)
        kernel_size = bpn["ks2"]
    else:
        if bpn["ks_ss"] != bpn["ks3"]:
            kw, mask = bpn["mlp_ss"](cam_idx, pos_enc, bpn_input, iteration)
            kernel_size = bpn["ks_ss"]
        else:
            kw, mask = bpn["mlp_ms"](cam_idx, pos_enc, bpn_input, iteration)
            kernel_size = bpn["ks3"]

    rgb = apply_kernel_chunked(shuffle_rgb, kw, kernel_size, bpn["blur_chunk_rows"])
    mask = mask[0]
    return mask * rgb + (1 - mask) * image, mask, kw


def apply_kernel_chunked(shuffle_rgb, kernel_weights, kernel_size, chunk_rows):
    """Apply a per-pixel kernel without materializing the full HxW unfold tensor."""
    _, _, height, width = shuffle_rgb.shape
    pad = kernel_size // 2
    padded = F.pad(shuffle_rgb, (pad, pad, pad, pad))
    chunks = []

    for y0 in range(0, height, chunk_rows):
        y1 = min(y0 + chunk_rows, height)
        local = padded[:, :, y0:y1 + 2 * pad, :]
        patches = F.unfold(local, kernel_size=(kernel_size, kernel_size), padding=0)
        patches = patches.view(1, 3, kernel_size ** 2, y1 - y0, width)
        kw = kernel_weights[:, :, y0:y1, :]
        chunks.append(torch.sum(patches * kw.unsqueeze(1), 2))

    return torch.cat(chunks, dim=2)[0]


def render_with_bpn(viewpoint_cam, triangles, pipe, bg, bpn, iteration):
    render_pkg = render(viewpoint_cam, triangles, pipe, bg)
    image = render_pkg["render"]
    if bpn.get("skip_bpn", False):
        render_pkg.update({
            "bpn_render": image,
            "bpn_mask": torch.zeros_like(image[:1]),
            "bpn_kernel_weights": None,
        })
        return render_pkg
    depth = render_pkg.get("surf_depth", torch.zeros_like(image[:1]))
    cam_idx = bpn["cam_index_by_name"].get(viewpoint_cam.image_name, 0)
    blur_image, mask, kernel_weights = apply_bpn_blur(image, depth, cam_idx, iteration, bpn)
    render_pkg.update({
        "bpn_render": blur_image,
        "bpn_mask": mask,
        "bpn_kernel_weights": kernel_weights,
    })
    return render_pkg


def training(
        dataset,
        opt,
        pipe,
        no_dome,
        outdoor,
        testing_iterations,
        save_iterations,
        checkpoint,
        debug_from,
        checkpoint_iterations=None,
        args=None,
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points, opt.set_sigma, no_dome)

    # BAGS-compatible LR mechanisms (--bags_lr_compat flag)
    bags_lr_compat = getattr(args, 'bags_lr_compat', False)
    _ms_steps = 6000  # iter at which BPN switches to full-scale kernel; reset position LR here
    _lr_scale = 1.0
    if bags_lr_compat:
        _ks3 = getattr(args, 'kernel_size3', 21)
        _ref_kernel = 17.0
        _lr_scale = (_ref_kernel / max(_ks3, _ref_kernel)) ** 0.5
        # Scale position LR before scheduler is built inside training_setup
        # (lr_final is auto-computed as lr_init/100 inside triangle_model, so only init needed)
        opt.lr_triangles_points_init *= _lr_scale
        print(f"[bags_lr_compat] ks3={_ks3}, lr_scale={_lr_scale:.4f}, ms_steps={_ms_steps}")

    triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma, opt.lr_triangles_points_init)

    # Also scale current optimizer param group LRs to match scaled scheduler init
    if bags_lr_compat and _lr_scale != 1.0:
        for pg in triangles.optimizer.param_groups:
            pg['lr'] *= _lr_scale

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        # Load full geometry from the Stage1 saved point cloud before restoring optimizer state.
        # The chkpnt.pth only saves features/opacity, not triangles_points/_mask/_sigma,
        # so without this the mask size stays at init while opacity size is from Stage1.
        ckpt_dir = os.path.dirname(checkpoint)
        geo_iter_dir = os.path.join(ckpt_dir, "point_cloud", f"iteration_{first_iter}")
        if os.path.exists(geo_iter_dir):
            print(f"[checkpoint] loading geometry from {geo_iter_dir}")
            triangles.load(geo_iter_dir)
            n = triangles.get_triangles_points.shape[0]
            triangles.triangle_area   = torch.zeros(n, device="cuda")
            triangles.image_size      = torch.zeros(n, device="cuda")
            triangles.importance_score = torch.zeros(n, device="cuda")
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ── BPN setup: reuse BAGS' estimated blur-kernel networks ────────────────
    bpn_ckpt = getattr(args, 'bpn_ckpt', None)
    freeze_bpn = getattr(args, 'freeze_bpn', True)
    ks1 = getattr(args, 'kernel_size1', 5)
    ks2 = getattr(args, 'kernel_size2', 9)
    ks3 = getattr(args, 'kernel_size3', 21)
    ks_ss = getattr(args, 'kernel_size_ss', ks3)
    train_cameras = scene.getTrainCameras()
    n_cams = len(train_cameras)
    cam_index_by_name = {cam.image_name: idx for idx, cam in enumerate(train_cameras)}
    h, w = int(scene.orig_h), int(scene.orig_w)
    if bpn_ckpt and dataset.resolution != 1:
        print(f"[BPN] warning: reusing BAGS blur kernels at render resolution -r {dataset.resolution}. "
              "Use -r 1 for strict apples-to-apples experiments.")

    mlp_ms, mlp_ss = build_bpn_modules(n_cams, h, w, ks1, ks2, ks3, ks_ss, args)
    mlp_ms, mlp_ss = load_bags_bpn_checkpoint(
        bpn_ckpt, mlp_ms, mlp_ss, n_cams, h, w, ks1, ks2, ks3, ks_ss, args
    )

    for p in mlp_ms.parameters(): p.requires_grad_(not freeze_bpn)
    for p in mlp_ss.parameters(): p.requires_grad_(not freeze_bpn)

    # BPN optimizer (only if not frozen)
    if not freeze_bpn:
        _bpn_lr_k = getattr(args, 'bpn_lr_kernel', 1e-4)
        _bpn_lr_m = getattr(args, 'bpn_lr_mask',   2.5e-4)
        _bpn_betas = tuple(getattr(args, 'bpn_betas', [0.9, 0.99]))
        _bpn_grad_clip = getattr(args, 'bpn_grad_clip', 0.0)  # 0 = no clip
        # Split: kernel heads (mlp_head*) vs mask heads (mlp_mask*) vs backbone
        def _split_bpn_params(module):
            k, m, b = [], [], []
            for n, p in module.named_parameters():
                if 'head' in n:   k.append(p)
                elif 'mask' in n: m.append(p)
                else:             b.append(p)
            return k, m, b
        k_ms, m_ms, b_ms = _split_bpn_params(mlp_ms)
        k_ss, m_ss, b_ss = _split_bpn_params(mlp_ss)
        bpn_optimizer = torch.optim.Adam([
            {'params': b_ms + b_ss, 'lr': _bpn_lr_k, 'name': 'bpn_base'},
            {'params': k_ms + k_ss, 'lr': _bpn_lr_k, 'name': 'bpn_kernel'},
            {'params': m_ms + m_ss, 'lr': _bpn_lr_m, 'name': 'bpn_mask'},
        ], betas=_bpn_betas, eps=1e-15)
        print(f"[BPN opt] lr_kernel={_bpn_lr_k} lr_mask={_bpn_lr_m} betas={_bpn_betas} grad_clip={_bpn_grad_clip}")
    else:
        bpn_optimizer = None
        _bpn_grad_clip = 0.0

    # NIMA per-frame weights
    nima_weights = load_nima_weights(getattr(args, 'nima_weights_path', None), dataset.source_path)

    # unfold operators for BPN
    unfold1 = torch.nn.Unfold(kernel_size=(ks1,ks1), padding=ks1//2).cuda()
    unfold2 = torch.nn.Unfold(kernel_size=(ks2,ks2), padding=ks2//2).cuda()
    unfold3 = torch.nn.Unfold(kernel_size=(ks3,ks3), padding=ks3//2).cuda()
    unfold_ss = (torch.nn.Unfold(kernel_size=(ks_ss,ks_ss), padding=ks_ss//2).cuda()
                 if ks_ss != ks3 else unfold3)
    # skip BPN entirely when frozen with no checkpoint (random weights would corrupt renders)
    skip_bpn = freeze_bpn and not bpn_ckpt

    bpn = {
        "mlp_ms": mlp_ms,
        "mlp_ss": mlp_ss,
        "unfold1": unfold1,
        "unfold2": unfold2,
        "unfold3": unfold3,
        "unfold_ss": unfold_ss,
        "ks1": ks1,
        "ks2": ks2,
        "ks3": ks3,
        "ks_ss": ks_ss,
        "blur_chunk_rows": 96,
        "cam_index_by_name": cam_index_by_name,
        "skip_bpn": skip_bpn,
        "no_curriculum": getattr(args, 'no_bpn_curriculum', False),
    }
    gt_depth_cache = load_gt_depth_cache(scene, dataset.source_path)
    lpips_fn = lpips.LPIPS(net='vgg').cpu()  # keep on CPU during training to save ~500 MB VRAM

    print(f"[BPN] n_cams={n_cams} image={h}x{w} ks={ks1}/{ks2}/{ks3} ss={ks_ss} freeze={freeze_bpn} skip={skip_bpn}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    total_dead = 0

    opacity_now = True

    new_round = False
    removed_them = False

    large_scene = triangles.large

    if large_scene and outdoor:
        loss_fn = l2_loss
    else:
        loss_fn = l1_loss

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # BAGS-compat: reset position LR scheduler after ms_steps (BPN switches to large kernel)
        _eff_lr_iter = (iteration - _ms_steps) if (bags_lr_compat and iteration > _ms_steps) else iteration
        triangles.update_learning_rate(_eff_lr_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_with_bpn(viewpoint_cam, triangles, pipe, bg, bpn, iteration)
        image = render_pkg["render"]

        # largest distance from point to center of image
        triangle_area = render_pkg["density_factor"].detach()
        # largest distance from point after applying sigma to center of image
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        # ── BPN: apply blur kernel to rendered image ─────────────────────────
        gt_image = viewpoint_cam.original_image.cuda()
        blur_image = render_pkg["bpn_render"]
        mask = render_pkg["bpn_mask"]

        # NIMA per-frame weight
        nima_w = nima_weights.get(viewpoint_cam.image_name, 1.0)

        pixel_loss = loss_fn(blur_image, gt_image)
        loss_image = nima_w * ((1.0 - opt.lambda_dssim) * pixel_loss +
                                opt.lambda_dssim * (1.0 - ssim(blur_image, gt_image)))

        # loss opacity
        loss_opacity = torch.abs(triangles.get_opacity).mean() * args.lambda_opacity

        # loss normal and distortion
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0 # 0.001
        rend_dist = render_pkg["rend_dist"]
        dist_loss = lambda_dist * (rend_dist).mean()
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean() 
        loss_size = loss_size * opt.lambda_size


        # ── GT depth supervision ──────────────────────────────────────────────
        loss_depth_gt = torch.tensor(0.0, device="cuda")
        if iteration > 250 and getattr(args, 'use_depth_loss', False):
            surf_d = render_pkg.get("surf_depth", None)
            gt_sensor = gt_depth_cache.get(viewpoint_cam.image_name)
            if surf_d is not None and gt_sensor is not None:
                sd = surf_d
                if sd.dim() == 2:
                    sd = sd.unsqueeze(0).unsqueeze(0)
                elif sd.dim() == 3:
                    sd = sd.unsqueeze(0)
                gt_d = F.interpolate(gt_sensor.cuda(), size=sd.shape[-2:], mode="nearest")
                depth_alpha = getattr(args, 'depth_loss_alpha', 0.01)
                if getattr(args, 'depth_scale_invariant', False):
                    # Scale-invariant: normalize both to [0,1] — for relative pseudo-depth
                    def _norm01(t):
                        mn, mx = t.min(), t.max()
                        return (t - mn) / (mx - mn + 1e-8)
                    loss_depth_gt = depth_alpha * F.l1_loss(_norm01(sd), _norm01(gt_d))
                else:
                    valid = (gt_d > 0.01).float()
                    loss_depth_gt = depth_alpha * ((torch.abs(sd - gt_d) * valid).sum() / (valid.sum() + 1e-6))

        # BPN mask loss: penalise large blend masks (encourages sparse blur)
        mask_loss_alpha = getattr(args, 'mask_loss_alpha', 0.0)
        loss_mask = mask_loss_alpha * mask.mean() if mask_loss_alpha > 0 else 0.0

        if iteration < opt.densify_until_iter:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_size + loss_depth_gt + loss_mask
        else:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_depth_gt + loss_mask

        loss.backward()
        if bpn_optimizer is not None:
            if _bpn_grad_clip > 0:
                all_bpn = [p for g in bpn_optimizer.param_groups for p in g['params']]
                torch.nn.utils.clip_grad_norm_(all_bpn, _bpn_grad_clip)
            bpn_optimizer.step()
            bpn_optimizer.zero_grad()

        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, background, bpn, lpips_fn)
            if iteration in save_iterations:
                print("\n[ITER {}] Saving Triangles".format(iteration))
                scene.save(iteration)
            if checkpoint_iterations and iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((triangles.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration % 1000 == 0:
                total_dead = 0

            if iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0 and iteration > opt.densify_from_iter:
                
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()

                if iteration > 1000 and not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                    
                    if not outdoor:
                        mask_test = triangles.image_size > 1400
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                          

                total_dead += dead_mask.sum()

                if opt.proba_distr == 0:
                    oddGroup = True
                elif opt.proba_distr == 1:
                    oddGroup = False
                else:
                    if opacity_now:
                        oddGroup = opacity_now
                        opacity_now = False
                    else:
                        oddGroup = opacity_now
                        opacity_now = True

                removed_them = True
                new_round = False

                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=oddGroup, dead_mask=dead_mask)


            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()


                if not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                triangles.remove_final_points(dead_mask)
                removed_them = True
                new_round = False

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                
    print("Training is done")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, pipe, background, bpn, lpips_fn):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        lpips_fn.cuda()  # move to GPU only for eval
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(render_with_bpn(viewpoint, scene.triangles, pipe, background, bpn, iteration)["bpn_render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.triangles.get_triangles_points.shape[0], iteration)
        lpips_fn.cpu()  # move back to CPU after eval
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    # BPN arguments
    parser.add_argument("--bpn_ckpt",          type=str, default=None,
                        help="Path to BAGS checkpoint to load BPN weights")
    parser.set_defaults(freeze_bpn=True)
    parser.add_argument("--freeze_bpn",         action="store_true", dest="freeze_bpn",
                        help="Freeze BPN and reuse the BAGS-estimated blur kernels (default)")
    parser.add_argument("--train_bpn",          action="store_false", dest="freeze_bpn",
                        help="Fine-tune the BPN online instead of freezing it")
    parser.add_argument("--kernel_size1",       type=int, default=5)
    parser.add_argument("--kernel_size2",       type=int, default=9)
    parser.add_argument("--kernel_size3",       type=int, default=21)
    parser.add_argument("--kernel_size_ss",     type=int, default=21)
    parser.add_argument("--not_use_rgbd",       action="store_true", default=False)
    parser.add_argument("--not_use_pe",         action="store_true", default=False)
    parser.add_argument("--nima_weights_path",  type=str, default=None,
                        help="Path to nima_weights_full.json for per-frame weighting")
    parser.add_argument("--use_depth_loss",        action="store_true", default=False,
                        help="Enable GT depth L1 supervision against depth maps in <scene>/depth/")
    parser.add_argument("--depth_loss_alpha",      type=float, default=0.01,
                        help="Weight for depth loss")
    parser.add_argument("--depth_scale_invariant", action="store_true", default=False,
                        help="Normalize both rendered and GT depth to [0,1] before L1 (for relative pseudo-depth)")
    parser.add_argument("--mask_loss_alpha",      type=float, default=0.0,
                        help="Weight for BPN mask loss (encourages sparse blur masks, same as BAGS default 0.001)")
    parser.add_argument("--bags_lr_compat",     action="store_true", default=False,
                        help="BAGS-compatible LR: kernel-size position-LR scaling + ms_steps LR reset at iter 6000")
    parser.add_argument("--no_bpn_curriculum",  action="store_true", default=False,
                        help="Skip progressive kernel curriculum: use large kernel (ks3) directly from warmup end")
    parser.add_argument("--bpn_lr_kernel",      type=float, default=1e-4,
                        help="LR for BPN kernel heads (mlp_head*)")
    parser.add_argument("--bpn_lr_mask",        type=float, default=2.5e-4,
                        help="LR for BPN mask/blend heads (mlp_mask*)")
    parser.add_argument("--bpn_betas",          nargs=2, type=float, default=[0.9, 0.99],
                        help="Adam betas for BPN optimizer")
    parser.add_argument("--bpn_grad_clip",      type=float, default=0.0,
                        help="Gradient clip norm for BPN (0=disabled)")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.no_dome,
             args.outdoor,
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             checkpoint_iterations=args.checkpoint_iterations,
             args=args,
             )
    
    # All done
    print("\nTraining complete.")
