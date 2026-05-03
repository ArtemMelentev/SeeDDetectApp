"""Seed segmentation pipeline in library + CLI modes.

Primary entrypoint for app integration:
    analyze_image(input_path, output_dir) -> dict
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_MAX_PIXELS = 12_000_000
DEFAULT_MAX_SIDE = 4096

# v2 pipeline defaults (see AlgorithmVersions/VersionWithoutScan/run_pipeline_v2.py)
DEFAULT_DO_SHADING = True
DEFAULT_TARGET_SHORT = 2400


# ──────────────────────────────────────────────────────────────────────
#  Legacy scan segmentation (kept for reference/backward compatibility)
# ──────────────────────────────────────────────────────────────────────

def _segment_all_seeds_legacy(img, edge_margin=10, dilation_k=51, diff_thresh=15):
    """
    Detect every seed on a white scan background.

    Uses morphological dilation to estimate local background brightness,
    then flags pixels noticeably darker than that background.
    A saturation filter rejects colorless bright pixels (pure-white gaps).
    Bright boundary removal surgically strips inter-seed gap leakage.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[edge_margin:h - edge_margin, edge_margin:w - edge_margin] = 255

    # Local background estimation
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_k, dilation_k))
    local_bg = cv2.dilate(gray, k).astype(np.float32)
    diff = local_bg - gray.astype(np.float32)

    seed_raw = ((diff > diff_thresh) & (roi > 0)).astype(np.uint8) * 255

    # Saturation filter: pure-white gaps are colorless (S < 8) and bright (gray > 200).
    # Real seeds always carry some hue (beige, brown, grey-striped).
    seed_raw[(sat < 8) & (gray > 200)] = 0

    # Secondary: adaptive threshold catches subtle edges of light seeds
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adapt = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=61, C=8,
    )
    adapt = cv2.bitwise_and(adapt, roi)

    k_dilate_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    diff_dilated = cv2.dilate(seed_raw, k_dilate_bridge)
    adapt_bridged = cv2.bitwise_and(adapt, diff_dilated)

    combined = cv2.bitwise_or(seed_raw, adapt_bridged)

    # Morphological cleanup — small close fills 1-2 px texture gaps within seeds
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open, iterations=1)

    # Remove bright boundary pixels — inter-seed gaps that still leaked through.
    # Interior bright pixels (specular highlights on seeds) are preserved.
    BRIGHT_BOUNDARY_THRESH = 225
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior = cv2.erode(cleaned, k_erode)
    bright_boundary = (gray > BRIGHT_BOUNDARY_THRESH) & (cleaned > 0) & (interior == 0)
    cleaned[bright_boundary] = 0
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open, iterations=1)

    # Remove tiny noise blobs and faint background artifacts
    min_area = max(60, h * w * 0.00004)
    n_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(cleaned)
    remove_mask = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, n_labels):
        area = comp_stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            remove_mask[labels == lbl] = 255
            continue
        comp_pixels = (labels == lbl)
        mean_gray = float(gray[comp_pixels].mean())
        if mean_gray > 220:
            remove_mask[labels == lbl] = 255
    result = cv2.bitwise_and(cleaned, cv2.bitwise_not(remove_mask))

    # Contours for visualization only
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return result, contours, {
        "seed_count": len(contours),
        "all_seed_area_px": int(np.sum(result > 0)),
        "image_area_px": h * w,
    }


# ──────────────────────────────────────────────────────────────────────
#  Pass 2 — black (unshelled / partially unshelled) seeds only
# ──────────────────────────────────────────────────────────────────────

def _segment_black_seeds_legacy(img, all_mask=None, edge_margin=10):
    """
    Detect dark / unshelled seeds (including partially peeled ones).

    Core-grow strategy with shadow rejection:
      1. Very dark core pixels (L < 75), cleaned gently
      2. Grow into semi-dark zone (L < 95), restricted to all_mask
      3. Trim boundary pixels brighter than L=85
      4. Reject components with low fill ratio (shadow sprawl)
         and insufficient dark-core fraction
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[edge_margin:h - edge_margin, edge_margin:w - edge_margin] = 255

    L_CORE = 75
    L_SEMI = 95

    # Stronger opening on core removes thin inter-seed shadow lines
    core = ((l_ch < L_CORE) & (roi > 0)).astype(np.uint8) * 255
    k_open_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core_clean = cv2.morphologyEx(core, cv2.MORPH_OPEN, k_open_core, iterations=1)

    semi_constraint = (l_ch < L_SEMI) & (roi > 0)
    if all_mask is not None:
        semi_constraint = semi_constraint & (all_mask > 0)
    semi = semi_constraint.astype(np.uint8) * 255

    grown = core_clean.copy()
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(15):
        expanded = cv2.dilate(grown, k_grow, iterations=1)
        expanded = cv2.bitwise_and(expanded, semi)
        if np.array_equal(expanded, grown):
            break
        grown = expanded

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Trim boundary pixels brighter than L=85 (shadow fringes)
    k_erode_blk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blk_interior = cv2.erode(grown, k_erode_blk)
    bright_blk_edge = (l_ch > 85) & (grown > 0) & (blk_interior == 0)
    grown[bright_blk_edge] = 0
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, k_open_sm, iterations=1)

    CORE_MIN = 0.15

    n_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(grown)
    result = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, n_labels):
        area = comp_stats[lbl, cv2.CC_STAT_AREA]
        comp_pixels = (labels == lbl)
        mean_l = float(l_ch[comp_pixels].mean())
        core_frac = float((l_ch[comp_pixels] < L_CORE).sum()) / area

        # Brightness-adaptive minimum area: dark seeds can be small,
        # lighter fragments need to be larger to be trustworthy
        if mean_l < 40:
            min_area = 100
        elif mean_l < 55:
            min_area = 200
        elif mean_l < 65:
            min_area = 350
        else:
            min_area = 500

        if area < min_area:
            continue
        if core_frac < CORE_MIN:
            continue

        blob_w = comp_stats[lbl, cv2.CC_STAT_WIDTH]
        blob_h = comp_stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = min(blob_w, blob_h) / max(blob_w, blob_h) if max(blob_w, blob_h) > 0 else 1
        fill = area / (blob_w * blob_h) if blob_w * blob_h > 0 else 0

        if mean_l < 65 and aspect < 0.06:
            continue
        if mean_l >= 65 and aspect < 0.10:
            continue

        # Reject shadow sprawl (low fill ratio = thin/scattered shape)
        if fill < 0.20:
            continue
        if mean_l > 55 and fill < 0.30:
            continue

        result[labels == lbl] = 255

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return result, contours, {
        "black_seed_count": len(contours),
        "black_seed_area_px": int(np.sum(result > 0)),
    }


# ──────────────────────────────────────────────────────────────────────
#  v2 helpers (paper-masked pipeline without GUI/CLI)
# ──────────────────────────────────────────────────────────────────────


def _odd(value: int | float, minimum: int = 3) -> int:
    out = max(minimum, int(round(value)))
    return out | 1


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    inv = cv2.bitwise_not(mask_u8)
    pad = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood_mask, (0, 0), 0)
    holes = pad[1:-1, 1:-1]
    return cv2.bitwise_or(mask_u8, holes)


def _largest_component(mask_u8: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8), 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == idx).astype(np.uint8) * 255
    return comp, int(stats[idx, cv2.CC_STAT_AREA])


def _component_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int]:
    pts = cv2.findNonZero(mask_u8)
    if pts is None:
        return 0, 0, mask_u8.shape[1], mask_u8.shape[0]
    return tuple(int(v) for v in cv2.boundingRect(pts))


def _detect_paper_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect paper as a bright low-saturation component.

    Ported from AlgorithmVersions/VersionWithoutScan/run_pipeline_v2.py.
    Returns full-frame mask if detection fails.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    work_short = 900
    short = min(orig_h, orig_w)
    scale = 1.0 if short <= work_short else work_short / float(short)
    if scale < 1.0:
        img_work = cv2.resize(
            img_bgr,
            (int(round(orig_w * scale)), int(round(orig_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        img_work = img_bgr
    h, w = img_work.shape[:2]
    diag = float(np.hypot(h, w))
    hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]

    attempts: list[tuple[int, int]] = [
        (25, 205),
        (35, 195),
        (45, 185),
        (55, 175),
        (70, 160),
        (85, 145),
        (100, 130),
        (115, 118),
    ]
    for s_q, g_q in ((45, 55), (55, 50), (65, 45), (75, 40)):
        attempts.append((int(np.percentile(sat, s_q)), int(np.percentile(gray, g_q))))

    close_k = _odd(diag * 0.018, 21)
    open_k = _odd(diag * 0.006, 7)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    candidates: list[tuple[np.ndarray, dict, float]] = []
    best: tuple[np.ndarray, dict] | None = None
    best_score = -1.0
    img_area = float(h * w)

    for sat_thr, gray_thr in attempts:
        sat_thr = int(np.clip(sat_thr, 20, 130))
        gray_thr = int(np.clip(gray_thr, 105, 215))
        raw = ((sat < sat_thr) & (gray > gray_thr)).astype(np.uint8) * 255
        if int(raw.sum()) == 0:
            continue
        mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        comp, area = _largest_component(mask)
        if area <= 0:
            continue
        frac = area / img_area
        x, y, bw, bh = _component_bbox(comp)
        bbox_area = max(1, bw * bh)
        fill = area / float(bbox_area)
        touches = int(x <= 2) + int(y <= 2) + int(x + bw >= w - 2) + int(y + bh >= h - 2)

        if frac < 0.08 or frac > 0.92:
            penalty = 0.25
        else:
            penalty = 0.0
        if fill < 0.35:
            penalty += 0.20
        if touches >= 3:
            penalty += 0.35
        score = frac + 0.15 * fill - penalty
        info = {
            "sat_thr": sat_thr,
            "gray_thr": gray_thr,
            "area_frac_raw": round(frac, 4),
            "bbox_fill_raw": round(fill, 4),
            "bbox_raw": [x, y, bw, bh],
            "border_touches_raw": touches,
        }
        candidates.append((comp, info, score))
        if score > best_score:
            best_score = score
            best = (comp, info)

    for comp, info, _ in candidates:
        frac = float(info["area_frac_raw"])
        fill = float(info["bbox_fill_raw"])
        touches = int(info["border_touches_raw"])
        if 0.62 <= frac <= 0.86 and fill >= 0.88 and touches <= 2:
            best = (comp, info)
            break

    if best is None:
        mask = np.ones((orig_h, orig_w), dtype=np.uint8) * 255
        return mask, {"fallback": "full_frame"}

    comp, info = best
    comp = _fill_holes(comp)
    comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, open_kernel, iterations=1)
    comp = _fill_holes(comp)

    erode_k = _odd(diag * 0.0015, 3)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    eroded = cv2.erode(comp, erode_kernel, iterations=1)
    if int((eroded > 0).sum()) > int((comp > 0).sum() * 0.90):
        comp = eroded

    if scale < 1.0:
        comp = cv2.resize(comp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        comp = ((comp > 0).astype(np.uint8)) * 255

    x, y, bw, bh = _component_bbox(comp)
    info.update({
        "area_frac": round(float((comp > 0).sum()) / float(orig_h * orig_w), 4),
        "bbox": [x, y, bw, bh],
        "erode_k": erode_k,
        "work_scale": round(scale, 4),
    })
    return comp, info


def _crop_to_paper(img_bgr: np.ndarray, paper_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = _component_bbox(paper_mask)
    return img_bgr[y:y + h, x:x + w].copy(), paper_mask[y:y + h, x:x + w].copy(), (x, y, w, h)


def _resize_short(img: np.ndarray, target_short: int, interpolation: int) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= target_short:
        return img
    scale = target_short / float(short)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=interpolation)


def _estimate_paper_color(img_bgr: np.ndarray, paper_mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    valid = np.ones(gray.shape, dtype=bool) if paper_mask is None else paper_mask > 0
    if int(valid.sum()) < 200:
        valid = np.ones(gray.shape, dtype=bool)
    sat_vals = sat[valid]
    gray_vals = gray[valid]
    cand = valid & (sat <= np.percentile(sat_vals, 45)) & (gray >= np.percentile(gray_vals, 60))
    if int(cand.sum()) < 200:
        cand = valid & (gray >= np.percentile(gray_vals, 80))
    if int(cand.sum()) < 50:
        cand = valid
    return np.median(img_bgr[cand].reshape(-1, 3).astype(np.float32), axis=0)


def _paper_pixel_mask(img_bgr: np.ndarray, paper_mask: np.ndarray, paper_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(paper_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    paper_lab = paper_lab.astype(np.float32).reshape(3)
    d_e = np.linalg.norm(lab - paper_lab[None, None, :], axis=2)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid = paper_mask > 0
    if int(valid.sum()) < 200:
        valid = np.ones(gray.shape, dtype=bool)
    gray_valid = gray[valid]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    min_gray = max(100, int(np.percentile(gray_valid, 35)))
    mask = valid & (d_e < 22.0) & (sat < 80) & (val > min_gray)
    out = mask.astype(np.uint8) * 255
    k = _odd(min(img_bgr.shape[:2]) / 180.0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


def _fit_poly_flatfield(channel: np.ndarray, mask_u8: np.ndarray, degree: int = 3) -> np.ndarray:
    h, w = channel.shape
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 500:
        value = float(np.median(channel[mask_u8 > 0])) if int((mask_u8 > 0).sum()) else float(np.median(channel))
        return np.full((h, w), value, dtype=np.float32)
    x_n = (xs.astype(np.float64) / max(1, w - 1)) * 2.0 - 1.0
    y_n = (ys.astype(np.float64) / max(1, h - 1)) * 2.0 - 1.0
    z = channel[ys, xs].astype(np.float64)
    terms: list[tuple[int, int]] = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
    a = np.stack([(x_n ** i) * (y_n ** j) for i, j in terms], axis=1)
    ata = a.T @ a
    ata += np.eye(ata.shape[0]) * 1e-3 * np.trace(ata) / ata.shape[0]
    coef = np.linalg.solve(ata, a.T @ z)
    xg = np.linspace(-1, 1, w, dtype=np.float64)
    yg = np.linspace(-1, 1, h, dtype=np.float64)
    xx, yy = np.meshgrid(xg, yg)
    surf = np.zeros((h, w), dtype=np.float64)
    for c, (i, j) in zip(coef, terms):
        surf += c * (xx ** i) * (yy ** j)
    return surf.astype(np.float32)


def _estimate_background(img_bgr: np.ndarray, paper_mask: np.ndarray, work_short: int = 900) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    scale = 1.0 if short <= work_short else work_short / float(short)
    if scale < 1.0:
        small = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(paper_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        small = img_bgr
        small_mask = paper_mask

    paper_bgr = _estimate_paper_color(small, small_mask)
    clean_paper = _paper_pixel_mask(small, small_mask, paper_bgr)
    paper_area = max(1, int((small_mask > 0).sum()))
    coverage = float((clean_paper > 0).sum()) / paper_area

    if coverage < 0.025:
        filled = small.copy()
        filled[small_mask == 0] = paper_bgr.astype(np.uint8)
        k = _odd(min(small.shape[:2]) / 6.0, 31)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg_small = cv2.dilate(filled, kernel)
        bg_small = cv2.medianBlur(bg_small, min(31, _odd(k / 2.0, 3)))
        bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=k * 0.55, sigmaY=k * 0.55)
    else:
        bg_small = np.zeros_like(small, dtype=np.float32)
        for channel in range(3):
            bg_small[:, :, channel] = _fit_poly_flatfield(small[:, :, channel], clean_paper, degree=3)
        sigma = max(5.0, min(small.shape[:2]) * 0.02)
        bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=sigma, sigmaY=sigma)
        paper_ref = small[clean_paper > 0].reshape(-1, 3).astype(np.float32)
        paper_med = np.median(paper_ref, axis=0)
        for channel in range(3):
            bg_small[:, :, channel] = np.maximum(bg_small[:, :, channel], paper_med[channel] * 0.55)

    if scale < 1.0:
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = bg_small
    return np.maximum(bg.astype(np.float32), 1.0)


def _paper_wb_and_levels(img_bgr: np.ndarray, paper_mask: np.ndarray, target: float = 240.0) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid = paper_mask > 0
    if int(valid.sum()) < 200:
        valid = np.ones(gray.shape, dtype=bool)
    threshold = np.percentile(gray[valid], 75)
    bright = valid & (gray >= threshold)
    if int(bright.sum()) < 500:
        bright = valid & (gray >= np.percentile(gray[valid], 60))
    if int(bright.sum()) > 0:
        mean_bgr = out[bright].reshape(-1, 3).mean(axis=0)
        gains = np.clip(target / np.maximum(mean_bgr, 1.0), 1.0 / 1.15, 1.15)
        out *= gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def _normalize_scan(img_bgr: np.ndarray, paper_mask: np.ndarray, do_shading: bool = True) -> np.ndarray:
    src = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=7)
    if do_shading:
        bg = _estimate_background(src, paper_mask)
        out = np.clip(src.astype(np.float32) / bg * 240.0, 0, 255).astype(np.uint8)
    else:
        out = src
    return _paper_wb_and_levels(out, paper_mask, target=240.0)


def _compute_delta_e(img_bgr: np.ndarray, paper_mask: np.ndarray | None = None) -> tuple[np.ndarray, tuple[float, float, float]]:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid = np.ones(gray.shape, dtype=bool) if paper_mask is None else paper_mask > 0
    if int(valid.sum()) < 500:
        valid = np.ones(gray.shape, dtype=bool)
    paper = valid & (hsv[:, :, 1] < 25) & (gray > np.percentile(gray[valid], 65))
    if int(paper.sum()) < 500:
        paper = valid & (gray >= np.percentile(gray[valid], 80))
    if int(paper.sum()) < 100:
        paper = valid
    p_l = float(light[paper].mean())
    p_a = float(a[paper].mean())
    p_b = float(b[paper].mean())
    d_e = np.sqrt((light - p_l) ** 2 + (a - p_a) ** 2 + (b - p_b) ** 2)
    return d_e.astype(np.float32), (p_l, p_a, p_b)


def _white_balance_to_paper(img_bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    paper_bgr = np.maximum(_estimate_paper_color(img_bgr, paper_mask), 1.0)
    gains = np.clip(255.0 / paper_bgr, 0.92, 1.18)
    out = img_bgr.astype(np.float32) * gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def _segment_all_seeds_v2(img_bgr: np.ndarray, paper_mask: np.ndarray) -> tuple[np.ndarray, list, dict]:
    h, w = img_bgr.shape[:2]
    diag = float(np.hypot(h, w))
    short = min(h, w)
    k3 = _odd(diag / 1200.0, 3)
    k7 = _odd(diag / 500.0, 7)

    d_e, paper_lab = _compute_delta_e(img_bgr, paper_mask)
    k_bg = _odd(short / 12.0, 31)
    d_e_u8 = np.clip(d_e, 0, 255).astype(np.uint8)
    scale = min(1.0, 800.0 / short) if short > 800 else 1.0
    if scale < 1.0:
        small = cv2.resize(d_e_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        k_bg_s = _odd(k_bg * scale, 9)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg_s, k_bg_s))
        bg_small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel_s)
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg, k_bg))
        bg = cv2.morphologyEx(d_e_u8, cv2.MORPH_OPEN, kernel_bg)
    d_e_local = np.clip(d_e - bg.astype(np.float32), 0, None)

    vals = d_e_local[(d_e_local > 1.0) & (paper_mask > 0)]
    if vals.size > 1000:
        thr_otsu, _ = cv2.threshold(
            np.clip(vals, 0, 255).astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        thr = float(max(3.0, min(15.0, thr_otsu)))
    else:
        thr = 5.0

    edge_k = _odd(diag * 0.0012, 3)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_k, edge_k))
    allowed = cv2.erode(paper_mask, edge_kernel, iterations=1)
    fg = ((d_e_local > thr) & (allowed > 0)).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3, k3))
    cleaned = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k7, k7))
    interior = cv2.erode(cleaned, kernel_erode)
    paper_like = d_e < 3.0
    bright_boundary = paper_like & (cleaned > 0) & (interior == 0)
    cleaned[bright_boundary] = 0
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    dark_pix = light < float(paper_lab[0]) - 8.0
    min_area = max(80, int(h * w * 5e-5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    result = np.zeros_like(cleaned)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        if float(d_e_local[comp].mean()) < max(2.0, thr * 0.8):
            continue
        dark_frac = float(dark_pix[comp].sum()) / float(area)
        if dark_frac < 0.45:
            continue
        result[comp] = 255

    dark_u8 = dark_pix.astype(np.uint8) * 255
    k_small = _odd(k3 / 2.0, 3)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    dark_closed = cv2.morphologyEx(dark_u8, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    dark_dilated = cv2.dilate(dark_closed, kernel_small, iterations=1)
    result = cv2.bitwise_and(result, dark_dilated)
    result = cv2.bitwise_and(result, paper_mask)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats_out = {
        "seed_count": len(contours),
        "all_seed_area_px": int((result > 0).sum()),
        "image_area_px": h * w,
    }
    return result, contours, stats_out


def _segment_black_seeds_v2(img_bgr: np.ndarray, all_mask: np.ndarray, paper_mask: np.ndarray) -> tuple[np.ndarray, list, dict]:
    h, w = img_bgr.shape[:2]
    diag = float(np.hypot(h, w))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    valid = (all_mask > 0) & (paper_mask > 0)

    l_core = 75.0
    l_semi = 95.0
    if int(valid.sum()) > 500:
        in_mask = light[valid]
        dark = in_mask[in_mask < 100]
        if dark.size > 500 and dark.size / max(1, in_mask.size) > 0.01:
            l_core = float(np.clip(np.percentile(dark, 90) + 5.0, 65.0, 85.0))
            l_semi = float(np.clip(l_core + 20.0, 85.0, 105.0))

    k3 = _odd(diag / 1200.0, 3)
    k5 = _odd(diag / 700.0, 5)
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k5, k5))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3, k3))
    core = ((light < l_core) & valid).astype(np.uint8) * 255
    core_clean = cv2.morphologyEx(core, cv2.MORPH_OPEN, kernel_core, iterations=1)
    core_clean = cv2.morphologyEx(core_clean, cv2.MORPH_OPEN, kernel_small, iterations=1)

    semi = ((light < l_semi) & valid).astype(np.uint8) * 255
    grown = core_clean.copy()
    kernel_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k5, k5))
    for _ in range(15):
        expanded = cv2.dilate(grown, kernel_grow, iterations=1)
        expanded = cv2.bitwise_and(expanded, semi)
        if np.array_equal(expanded, grown):
            break
        grown = expanded

    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, kernel_grow, iterations=1)
    interior = cv2.erode(grown, kernel_small)
    bright_edge = (light > l_core + 12.0) & (grown > 0) & (interior == 0)
    grown[bright_edge] = 0
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, kernel_small, iterations=1)

    base_min = max(60, int(h * w * 3.0e-5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    result = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp = labels == label
        mean_l = float(light[comp].mean())
        if mean_l < 40.0:
            min_area = base_min
        elif mean_l < 55.0:
            min_area = base_min * 2
        elif mean_l < 65.0:
            min_area = base_min * 4
        else:
            min_area = base_min * 6
        if area < min_area:
            continue
        core_frac = float((light[comp] < l_core).sum()) / float(area)
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(1, bw * bh))
        aspect = min(bw, bh) / float(max(1, max(bw, bh)))
        if core_frac < 0.15:
            continue
        if mean_l < 65.0 and aspect < 0.06:
            continue
        if mean_l >= 65.0 and aspect < 0.10:
            continue
        if fill < 0.18:
            continue
        if mean_l > 55.0 and fill < 0.30:
            continue
        result[comp] = 255

    result = cv2.bitwise_and(result, all_mask)
    result = cv2.bitwise_and(result, paper_mask)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats_out = {
        "black_seed_count": len(contours),
        "black_seed_area_px": int((result > 0).sum()),
    }
    return result, contours, stats_out


def _recover_black_pixels(img_bgr: np.ndarray, all_mask: np.ndarray, paper_mask: np.ndarray, light: np.ndarray | None = None) -> np.ndarray:
    if light is None:
        light = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    h, w = all_mask.shape
    diag = float(np.hypot(h, w))
    valid = (all_mask > 0) & (paper_mask > 0)
    hard = ((light < 55.0) & valid).astype(np.uint8) * 255
    k3 = _odd(diag / 1200.0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3, k3))
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, kernel, iterations=1)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    out = np.zeros_like(hard)
    min_area = max(150, int(h * w * 4.0e-5))
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        l_in = light[comp]
        mean_l = float(l_in.mean())
        std_l = float(l_in.std())
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(1, bw * bh))
        aspect = min(bw, bh) / float(max(1, max(bw, bh)))
        if mean_l > 67.0:
            continue
        if std_l < 6.0:
            continue
        if fill < 0.30:
            continue
        if aspect < 0.18:
            continue
        comp_u8 = comp.astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < 0.62:
            continue
        hard_frac = float((l_in < 55.0).sum()) / float(area)
        if hard_frac < 0.25:
            continue
        out[comp] = 255
    return out


def _interior_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    inv = cv2.bitwise_not(mask_u8)
    pad = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood_mask, (0, 0), 0)
    return pad[1:-1, 1:-1]


def _fixup_masks(img_bgr: np.ndarray, paper_mask: np.ndarray, all_mask: np.ndarray, black_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    paper_pix = (paper_mask > 0) & (all_mask == 0) & (light > 190) & (chroma < 15)
    paper_l = float(np.median(light[paper_pix])) if int(paper_pix.sum()) > 1000 else 240.0

    holes = _interior_holes(all_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    accepted = np.zeros_like(holes)
    for label in range(1, n):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        if int(((paper_mask > 0) & comp).sum()) < area * 0.95:
            continue
        med_l = float(np.median(light[comp]))
        mean_chroma = float(chroma[comp].mean())
        if mean_chroma >= 45.0:
            continue
        if area < 200 and med_l < paper_l - 20.0:
            accepted[comp] = 255
        elif area < 2500 and med_l < paper_l - 45.0:
            accepted[comp] = 255
        elif med_l < paper_l - 80.0:
            accepted[comp] = 255

    fixed_all = cv2.bitwise_or(all_mask, accepted)
    fixed_all = cv2.bitwise_and(fixed_all, paper_mask)

    recovered = _recover_black_pixels(img_bgr, fixed_all, paper_mask, light)
    fixed_black = cv2.bitwise_or(black_mask, recovered)
    fixed_black = cv2.bitwise_and(fixed_black, fixed_all)
    fixed_black = cv2.bitwise_and(fixed_black, paper_mask)
    return fixed_all, fixed_black


def _overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    vis = img_bgr.copy()
    overlay = vis.copy()
    overlay[mask_u8 > 0] = color
    cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
    return vis


def _visualize_all_v2(img_bgr: np.ndarray, all_mask: np.ndarray, seed_count: int, all_area_px: int) -> np.ndarray:
    vis = _overlay_mask(img_bgr, all_mask, (0, 210, 0), 0.35)
    contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    _put_text_block(vis, [f"All seeds: {seed_count}", f"Area: {all_area_px} px"], origin=(12, 30), font_scale=0.7, thickness=2)
    return vis


def _visualize_black_v2(img_bgr: np.ndarray, black_mask: np.ndarray, black_count: int, black_area_px: int) -> np.ndarray:
    vis = _overlay_mask(img_bgr, black_mask, (0, 0, 230), 0.42)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    _put_text_block(vis, [f"Black seeds: {black_count}", f"Area: {black_area_px} px"], origin=(12, 30), font_scale=0.7, thickness=2)
    return vis


# ──────────────────────────────────────────────────────────────────────
#  Visualization helpers
# ──────────────────────────────────────────────────────────────────────

def _put_text_block(vis, lines, origin=(12, 30), font_scale=0.7, thickness=2):
    y = origin[1]
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(vis, (origin[0] - 4, y - th - 4), (origin[0] + tw + 4, y + 4), (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        y += th + 14


def visualize_all_seeds(img, mask, contours, stats, path):
    vis = img.copy()
    overlay = img.copy()
    overlay[mask > 0] = (0, 200, 0)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    _put_text_block(vis, [
        f"All seeds: {stats['seed_count']}",
        f"Area: {stats['all_seed_area_px']} px",
    ])
    if not _safe_cv2_write(Path(path), vis, ".jpg", quality=95):
        raise OSError(f"Cannot write visualization file: {path}")


def visualize_black_seeds(img, mask, contours, stats, path):
    vis = img.copy()
    overlay = img.copy()
    overlay[mask > 0] = (0, 0, 220)
    cv2.addWeighted(overlay, 0.40, vis, 0.60, 0, vis)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(vis, str(i + 1), (cx - 6, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    _put_text_block(vis, [
        f"Black seeds: {stats['black_seed_count']}",
        f"Area: {stats['black_seed_area_px']} px",
    ])
    if not _safe_cv2_write(Path(path), vis, ".jpg", quality=95):
        raise OSError(f"Cannot write visualization file: {path}")


def _safe_cv2_read(path: Path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size > 0:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _safe_cv2_write(path: Path, image, ext: str, quality: int | None = None):
    params = []
    if quality is not None and ext.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def _resize_if_needed(img, max_pixels=DEFAULT_MAX_PIXELS, max_side=DEFAULT_MAX_SIDE):
    h, w = img.shape[:2]
    scale_px = math.sqrt(max_pixels / float(max(w * h, 1)))
    scale_side = max_side / float(max(w, h))
    scale = min(1.0, scale_px, scale_side)

    if scale >= 1.0:
        return img, {
            "resized": False,
            "scale": 1.0,
            "processed_w": w,
            "processed_h": h,
        }

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, {
        "resized": True,
        "scale": float(scale),
        "processed_w": new_w,
        "processed_h": new_h,
    }


def _error_payload(code: str, message: str, details=None):
    payload = {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return payload


def _safe_percent(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) * 100.0 / float(denominator)


def analyze_image(
    input_path,
    output_dir,
    max_pixels=DEFAULT_MAX_PIXELS,
    max_side=DEFAULT_MAX_SIDE,
    do_shading: bool = DEFAULT_DO_SHADING,
    target_short: int = DEFAULT_TARGET_SHORT,
):
    """Analyze one image and return a structured payload for UI/native bridges."""
    started = time.perf_counter()

    try:
        input_path = Path(input_path).expanduser()
        output_dir = Path(output_dir).expanduser()

        if not input_path.exists():
            return _error_payload("file_not_found", f"Input file not found: {input_path}")
        if not input_path.is_file():
            return _error_payload("invalid_input", f"Input path is not a file: {input_path}")
        if input_path.suffix.lower() not in SUPPORTED_EXTS:
            return _error_payload("unsupported_format", f"Unsupported file extension: {input_path.suffix}")

        input_path = input_path.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        img = _safe_cv2_read(input_path)
        if img is None:
            return _error_payload("decode_error", "Cannot decode image (file may be corrupted)")

        image_h, image_w = img.shape[:2]
        prepared_img, prep = _resize_if_needed(img, max_pixels=max_pixels, max_side=max_side)
        stem = input_path.stem

        try:
            # v2: detect paper -> crop -> (optional upscale) -> normalize -> segment -> fixup
            paper_full, _detect_info = _detect_paper_mask(prepared_img)
            crop_img, crop_paper, _crop = _crop_to_paper(prepared_img, paper_full)

            if target_short and int(target_short) > 0:
                crop_img = _resize_short(crop_img, int(target_short), cv2.INTER_LANCZOS4)
                crop_paper = _resize_short(crop_paper, int(target_short), cv2.INTER_NEAREST)
                crop_paper = ((crop_paper > 0).astype(np.uint8)) * 255

            normalized = _normalize_scan(crop_img, crop_paper, do_shading=bool(do_shading))
            normalized = _white_balance_to_paper(normalized, crop_paper)

            all_mask, _all_cnt, _all_stats = _segment_all_seeds_v2(normalized, crop_paper)
            blk_mask, _blk_cnt, _blk_stats = _segment_black_seeds_v2(normalized, all_mask, crop_paper)
            all_mask, blk_mask = _fixup_masks(normalized, crop_paper, all_mask, blk_mask)

            # Refresh counts/contours after fixup.
            all_cnt, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blk_cnt, _ = cv2.findContours(blk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_stats = {
                "seed_count": int(len(all_cnt)),
                "all_seed_area_px": int((all_mask > 0).sum()),
                "image_area_px": int(normalized.shape[0] * normalized.shape[1]),
            }
            blk_stats = {
                "black_seed_count": int(len(blk_cnt)),
                "black_seed_area_px": int((blk_mask > 0).sum()),
            }
        except cv2.error as exc:
            return _error_payload("opencv_error", "OpenCV processing failed", str(exc))

        all_seed_area_px = int(all_stats.get("all_seed_area_px", 0))
        black_seed_area_px = int(blk_stats.get("black_seed_area_px", 0))
        image_area_px = int(all_stats.get("image_area_px", 0))

        all_seed_ratio_pct = _safe_percent(all_seed_area_px, image_area_px)
        black_seed_ratio_pct = _safe_percent(black_seed_area_px, image_area_px)
        black_to_all_seed_ratio_pct = _safe_percent(black_seed_area_px, all_seed_area_px)

        all_mask_path = output_dir / f"{stem}_all_mask.png"
        all_overlay_path = output_dir / f"{stem}_all_overlay.jpg"
        black_mask_path = output_dir / f"{stem}_black_mask.png"
        black_overlay_path = output_dir / f"{stem}_black_overlay.jpg"

        if not _safe_cv2_write(all_mask_path, all_mask, ".png"):
            return _error_payload("write_error", f"Cannot write artifact: {all_mask_path}")
        try:
            all_vis = _visualize_all_v2(
                normalized,
                all_mask,
                seed_count=int(all_stats["seed_count"]),
                all_area_px=int(all_stats["all_seed_area_px"]),
            )
            if not _safe_cv2_write(all_overlay_path, all_vis, ".jpg", quality=92):
                raise OSError(f"Cannot write visualization file: {all_overlay_path}")
        except OSError as exc:
            return _error_payload("write_error", "Cannot write all-seed overlay", str(exc))

        if not _safe_cv2_write(black_mask_path, blk_mask, ".png"):
            return _error_payload("write_error", f"Cannot write artifact: {black_mask_path}")
        try:
            black_vis = _visualize_black_v2(
                normalized,
                blk_mask,
                black_count=int(blk_stats["black_seed_count"]),
                black_area_px=int(blk_stats["black_seed_area_px"]),
            )
            if not _safe_cv2_write(black_overlay_path, black_vis, ".jpg", quality=92):
                raise OSError(f"Cannot write visualization file: {black_overlay_path}")
        except OSError as exc:
            return _error_payload("write_error", "Cannot write black-seed overlay", str(exc))

        processing_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "image": str(input_path),
            "image_w": int(image_w),
            "image_h": int(image_h),
            "processing_ms": processing_ms,
            # Keep contract stable: only the existing metric keys are included.
            "seed_count": int(all_stats.get("seed_count", 0)),
            "all_seed_area_px": int(all_stats.get("all_seed_area_px", 0)),
            "image_area_px": int(all_stats.get("image_area_px", 0)),
            "black_seed_count": int(blk_stats.get("black_seed_count", 0)),
            "black_seed_area_px": int(blk_stats.get("black_seed_area_px", 0)),
            "all_seed_ratio_pct": all_seed_ratio_pct,
            "black_seed_ratio_pct": black_seed_ratio_pct,
            "black_to_all_seed_ratio_pct": black_to_all_seed_ratio_pct,
            "preprocessing": prep,
            "artifacts": {
                "all_mask": str(all_mask_path),
                "all_overlay": str(all_overlay_path),
                "black_mask": str(black_mask_path),
                "black_overlay": str(black_overlay_path),
            },
        }
    except cv2.error as exc:
        return _error_payload("opencv_error", "OpenCV processing failed", str(exc))
    except Exception as exc:  # pragma: no cover
        return _error_payload("unexpected_error", "Unexpected processing failure", str(exc))


def process_image(image_path, output_dir):
    """Backward-compatible adapter for legacy batch workflow."""
    result = analyze_image(image_path, output_dir)
    if not result.get("ok"):
        return None
    return {
        "image": result["image"],
        "image_w": result["image_w"],
        "image_h": result["image_h"],
        "seed_count": result["seed_count"],
        "all_seed_area_px": result["all_seed_area_px"],
        "black_seed_count": result["black_seed_count"],
        "black_seed_area_px": result["black_seed_area_px"],
        "all_seed_ratio_pct": result["all_seed_ratio_pct"],
        "black_seed_ratio_pct": result["black_seed_ratio_pct"],
        "black_to_all_seed_ratio_pct": result["black_to_all_seed_ratio_pct"],
        "processing_ms": result["processing_ms"],
    }


def _collect_images(path: Path):
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTS else []
    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith(".")
        )
    return []


def _build_parser():
    parser = argparse.ArgumentParser(description="Seed segmentation (single file or directory)")
    parser.add_argument("input", type=str, help="Input file or folder")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Max pixels before downscale")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="Max image side before downscale")
    return parser


def main():
    args = _build_parser().parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: input does not exist: {input_path}")
        return

    if args.output:
        output_dir = Path(args.output)
    elif input_path.is_dir():
        output_dir = input_path / "results"
    else:
        output_dir = input_path.parent / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    images = _collect_images(input_path)
    if not images:
        print("No supported images found")
        return

    all_results = []
    print(f"Found {len(images)} image(s)")

    for img_path in images:
        result = analyze_image(
            str(img_path),
            str(output_dir),
            max_pixels=args.max_pixels,
            max_side=args.max_side,
        )
        all_results.append(result)

        if result.get("ok"):
            print(
                f"[{img_path.name}] seeds={result['seed_count']} "
                f"black={result['black_seed_count']} ms={result['processing_ms']}"
            )
        else:
            err = result.get("error", {})
            print(f"[{img_path.name}] ERROR {err.get('code')}: {err.get('message')}")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
