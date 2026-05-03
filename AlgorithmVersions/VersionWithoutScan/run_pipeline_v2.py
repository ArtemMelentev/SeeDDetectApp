#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


Logger = Callable[[str], None]


def imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray, params: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(str(path))


def odd(value: int | float, minimum: int = 3) -> int:
    out = max(minimum, int(round(value)))
    return out | 1


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    inv = cv2.bitwise_not(mask_u8)
    pad = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood_mask, (0, 0), 0)
    holes = pad[1:-1, 1:-1]
    return cv2.bitwise_or(mask_u8, holes)


def largest_component(mask_u8: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8), 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == idx).astype(np.uint8) * 255
    return comp, int(stats[idx, cv2.CC_STAT_AREA])


def component_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int]:
    pts = cv2.findNonZero(mask_u8)
    if pts is None:
        return 0, 0, mask_u8.shape[1], mask_u8.shape[0]
    return tuple(int(v) for v in cv2.boundingRect(pts))


def detect_paper_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect the paper as one bright low-saturation component.

    No convex hull is used.  Convex hulls were the source of table triangles
    on partially visible or skewed sheets.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    work_short = 900
    short = min(orig_h, orig_w)
    scale = 1.0 if short <= work_short else work_short / float(short)
    if scale < 1.0:
        img_work = cv2.resize(img_bgr, (int(round(orig_w * scale)), int(round(orig_h * scale))), interpolation=cv2.INTER_AREA)
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
    # Add percentile-driven attempts for dim no-flash frames.
    for s_q, g_q in ((45, 55), (55, 50), (65, 45), (75, 40)):
        attempts.append((int(np.percentile(sat, s_q)), int(np.percentile(gray, g_q))))

    close_k = odd(diag * 0.018, 21)
    open_k = odd(diag * 0.006, 7)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    candidates: list[tuple[int, np.ndarray, dict, float]] = []
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
        comp, area = largest_component(mask)
        if area <= 0:
            continue
        frac = area / img_area
        x, y, bw, bh = component_bbox(comp)
        bbox_area = max(1, bw * bh)
        fill = area / float(bbox_area)
        touches = int(x <= 2) + int(y <= 2) + int(x + bw >= w - 2) + int(y + bh >= h - 2)

        # Paper should be a large but not whole-frame low-saturation object.
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
        candidates.append((len(candidates), comp, info, score))
        if score > best_score:
            best_score = score
            best = (comp, info)

    # Prefer the first strict candidate in threshold order.  On the target
    # photos this picks the actual bright paper rectangle and avoids the later
    # soft thresholds that merge dim table/background into the sheet.
    for _, comp, info, _ in candidates:
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
    comp = fill_holes(comp)
    comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, open_kernel, iterations=1)
    comp = fill_holes(comp)

    # A small inward shrink prevents table pixels on the true sheet edge from
    # becoming legal seed-mask pixels.  Keep it small to avoid cutting seeds.
    erode_k = odd(diag * 0.0015, 3)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    eroded = cv2.erode(comp, erode_kernel, iterations=1)
    if int((eroded > 0).sum()) > int((comp > 0).sum() * 0.90):
        comp = eroded

    if scale < 1.0:
        comp = cv2.resize(comp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        comp = ((comp > 0).astype(np.uint8)) * 255

    x, y, bw, bh = component_bbox(comp)
    info.update({
        "area_frac": round(float((comp > 0).sum()) / float(orig_h * orig_w), 4),
        "bbox": [x, y, bw, bh],
        "erode_k": erode_k,
        "work_scale": round(scale, 4),
    })
    return comp, info


def draw_paper_detect(img_bgr: np.ndarray, paper_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    vis = img_bgr.copy()
    overlay = vis.copy()
    overlay[paper_mask > 0] = (255, 180, 0)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 3)
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 3)
    return vis


def crop_to_paper(img_bgr: np.ndarray, paper_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = component_bbox(paper_mask)
    return img_bgr[y:y + h, x:x + w].copy(), paper_mask[y:y + h, x:x + w].copy(), (x, y, w, h)


def resize_short(img: np.ndarray, target_short: int, interpolation: int) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= target_short:
        return img
    scale = target_short / float(short)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=interpolation)


def estimate_paper_color(img_bgr: np.ndarray, paper_mask: np.ndarray | None = None) -> np.ndarray:
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


def paper_pixel_mask(img_bgr: np.ndarray, paper_mask: np.ndarray, paper_bgr: np.ndarray) -> np.ndarray:
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
    k = odd(min(img_bgr.shape[:2]) / 180.0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


def fit_poly_flatfield(channel: np.ndarray, mask_u8: np.ndarray, degree: int = 3) -> np.ndarray:
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


def estimate_background(img_bgr: np.ndarray, paper_mask: np.ndarray, work_short: int = 900) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    scale = 1.0 if short <= work_short else work_short / float(short)
    if scale < 1.0:
        small = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(paper_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        small = img_bgr
        small_mask = paper_mask

    paper_bgr = estimate_paper_color(small, small_mask)
    clean_paper = paper_pixel_mask(small, small_mask, paper_bgr)
    paper_area = max(1, int((small_mask > 0).sum()))
    coverage = float((clean_paper > 0).sum()) / paper_area

    if coverage < 0.025:
        filled = small.copy()
        filled[small_mask == 0] = paper_bgr.astype(np.uint8)
        k = odd(min(small.shape[:2]) / 6.0, 31)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg_small = cv2.dilate(filled, kernel)
        bg_small = cv2.medianBlur(bg_small, min(31, odd(k / 2.0, 3)))
        bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=k * 0.55, sigmaY=k * 0.55)
    else:
        bg_small = np.zeros_like(small, dtype=np.float32)
        for channel in range(3):
            bg_small[:, :, channel] = fit_poly_flatfield(small[:, :, channel], clean_paper, degree=3)
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


def paper_wb_and_levels(img_bgr: np.ndarray, paper_mask: np.ndarray, target: float = 240.0) -> np.ndarray:
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


def normalize_scan(img_bgr: np.ndarray, paper_mask: np.ndarray, do_shading: bool = True) -> np.ndarray:
    src = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=7)
    if do_shading:
        bg = estimate_background(src, paper_mask)
        out = np.clip(src.astype(np.float32) / bg * 240.0, 0, 255).astype(np.uint8)
    else:
        out = src
    return paper_wb_and_levels(out, paper_mask, target=240.0)


def compute_delta_e(img_bgr: np.ndarray, paper_mask: np.ndarray | None = None) -> tuple[np.ndarray, tuple[float, float, float]]:
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


def white_balance_to_paper(img_bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    paper_bgr = np.maximum(estimate_paper_color(img_bgr, paper_mask), 1.0)
    gains = np.clip(255.0 / paper_bgr, 0.92, 1.18)
    out = img_bgr.astype(np.float32) * gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def segment_all_seeds(img_bgr: np.ndarray, paper_mask: np.ndarray, return_debug: bool = False):
    h, w = img_bgr.shape[:2]
    diag = float(np.hypot(h, w))
    short = min(h, w)
    k3 = odd(diag / 1200.0, 3)
    k7 = odd(diag / 500.0, 7)

    d_e, paper_lab = compute_delta_e(img_bgr, paper_mask)
    k_bg = odd(short / 12.0, 31)
    d_e_u8 = np.clip(d_e, 0, 255).astype(np.uint8)
    scale = min(1.0, 800.0 / short) if short > 800 else 1.0
    if scale < 1.0:
        small = cv2.resize(d_e_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        k_bg_s = odd(k_bg * scale, 9)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg_s, k_bg_s))
        bg_small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel_s)
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg, k_bg))
        bg = cv2.morphologyEx(d_e_u8, cv2.MORPH_OPEN, kernel_bg)
    d_e_local = np.clip(d_e - bg.astype(np.float32), 0, None)

    vals = d_e_local[(d_e_local > 1.0) & (paper_mask > 0)]
    if vals.size > 1000:
        thr_otsu, _ = cv2.threshold(np.clip(vals, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(max(3.0, min(15.0, thr_otsu)))
    else:
        thr = 5.0

    edge_k = odd(diag * 0.0012, 3)
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
    k_small = odd(k3 / 2.0, 3)
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
        "deltaE_thresh_used": round(float(thr), 2),
        "paper_L": round(float(paper_lab[0]), 1),
        "paper_a": round(float(paper_lab[1]), 2),
        "paper_b": round(float(paper_lab[2]), 2),
        "method": "v2_paper_masked_tophat",
    }
    if return_debug:
        stats_out["_debug"] = {"dE": d_e, "dE_local": d_e_local}
    return result, contours, stats_out


def segment_black_seeds(img_bgr: np.ndarray, all_mask: np.ndarray, paper_mask: np.ndarray):
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

    k3 = odd(diag / 1200.0, 3)
    k5 = odd(diag / 700.0, 5)
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
        "L_core_used": round(float(l_core), 1),
        "L_semi_used": round(float(l_semi), 1),
    }
    return result, contours, stats_out


def recover_black_pixels(img_bgr: np.ndarray, all_mask: np.ndarray, paper_mask: np.ndarray, light: np.ndarray | None = None) -> np.ndarray:
    if light is None:
        light = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    h, w = all_mask.shape
    diag = float(np.hypot(h, w))
    valid = (all_mask > 0) & (paper_mask > 0)
    hard = ((light < 55.0) & valid).astype(np.uint8) * 255
    k3 = odd(diag / 1200.0, 3)
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
        med_l = float(np.median(l_in))
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
        if fill < 0.18 or aspect < 0.06:
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


def interior_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    inv = cv2.bitwise_not(mask_u8)
    pad = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood_mask, (0, 0), 0)
    return pad[1:-1, 1:-1]


def fixup_masks(img_bgr: np.ndarray, paper_mask: np.ndarray, all_mask: np.ndarray, black_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    paper_pix = (paper_mask > 0) & (all_mask == 0) & (light > 190) & (chroma < 15)
    paper_l = float(np.median(light[paper_pix])) if int(paper_pix.sum()) > 1000 else 240.0

    holes = interior_holes(all_mask)
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

    recovered = recover_black_pixels(img_bgr, fixed_all, paper_mask, light)
    fixed_black = cv2.bitwise_or(black_mask, recovered)
    fixed_black = cv2.bitwise_and(fixed_black, fixed_all)
    fixed_black = cv2.bitwise_and(fixed_black, paper_mask)
    info = {
        "paper_L": round(paper_l, 1),
        "all_added_px": int((fixed_all > 0).sum() - (all_mask > 0).sum()),
        "black_added_px": int((fixed_black > 0).sum() - (black_mask > 0).sum()),
    }
    return fixed_all, fixed_black, info


def put_text_block(vis: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 30)) -> None:
    scale = max(0.55, min(vis.shape[:2]) / 1800.0)
    thickness = max(1, int(round(scale * 2)))
    y = origin[1]
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(vis, (origin[0] - 4, y - th - 5), (origin[0] + tw + 4, y + 5), (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += th + 12


def overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    vis = img_bgr.copy()
    overlay = vis.copy()
    overlay[mask_u8 > 0] = color
    cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
    return vis


def visualize_all(img_bgr: np.ndarray, all_mask: np.ndarray, stats: dict) -> np.ndarray:
    vis = overlay_mask(img_bgr, all_mask, (0, 210, 0), 0.35)
    contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    put_text_block(vis, [
        f"All seeds: {stats['seed_count']}",
        f"Area: {stats['all_seed_area_px']}",
        f"dE thr: {stats['deltaE_thresh_used']}",
    ])
    return vis


def visualize_black(img_bgr: np.ndarray, black_mask: np.ndarray, stats: dict) -> np.ndarray:
    vis = overlay_mask(img_bgr, black_mask, (0, 0, 230), 0.42)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    put_text_block(vis, [
        f"Black seeds: {stats['black_seed_count']}",
        f"Area: {stats['black_seed_area_px']}",
        f"L core/semi: {stats['L_core_used']}/{stats['L_semi_used']}",
    ])
    return vis


def audit_result(img_bgr: np.ndarray, paper_mask: np.ndarray, all_mask: np.ndarray, black_mask: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    paper_pixels = (paper_mask > 0) & (all_mask == 0) & (light > 190) & (chroma < 15)
    paper_l = float(np.median(light[paper_pixels])) if int(paper_pixels.sum()) > 1000 else 240.0

    scope = paper_mask > 0
    nonpaper = scope & (all_mask == 0) & (black_mask == 0) & ((light < 170) | (chroma > 15))
    leak_blobs = connected_blob_summary(nonpaper, min_area=max(80, int(h * w * 1.0e-4)), light=light, chroma=chroma)

    dark_outside = scope & (all_mask == 0) & (light < paper_l - 25.0) & (chroma < 30)
    missed = connected_blob_summary(dark_outside, min_area=max(120, int(h * w * 4.0e-5)), light=light)

    very_dark = scope & (all_mask > 0) & (black_mask == 0) & (light < 65.0)
    missed_black = connected_blob_summary(very_dark, min_area=200, light=light)

    black_shadow_like = []
    n_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    for label in range(1, n_b):
        area = int(stats_b[label, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        comp = labels_b == label
        l_in = light[comp]
        med_l = float(np.median(l_in))
        std_l = float(np.std(l_in))
        flag = "ok"
        if med_l > 95.0 and std_l < 8.0:
            flag = "shadow_like"
        elif med_l > 110.0:
            flag = "too_bright"
        if flag != "ok":
            black_shadow_like.append({
                "area": area,
                "bbox": bbox_from_stats(stats_b, label),
                "med_L": round(med_l, 1),
                "std_L": round(std_l, 1),
                "flag": flag,
            })
    black_shadow_like.sort(key=lambda item: -item["area"])

    n_all, _, stats_all, _ = cv2.connectedComponentsWithStats(all_mask, connectivity=8)
    n_black, _, stats_black, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    return {
        "size": [w, h],
        "paper_L": round(paper_l, 1),
        "all_count": int(max(0, n_all - 1)),
        "all_area_px": int((all_mask > 0).sum()),
        "black_count": int(max(0, n_black - 1)),
        "black_area_px": int((black_mask > 0).sum()),
        "black_to_all_ratio": round(float((black_mask > 0).sum()) / max(1, int((all_mask > 0).sum())), 4),
        "leak_blobs": len(leak_blobs),
        "leak_area_px": int(sum(item["area"] for item in leak_blobs)),
        "leak_top": leak_blobs[:5],
        "missed_seed_blobs": len(missed),
        "missed_seed_area_px": int(sum(item["area"] for item in missed)),
        "missed_seed_top": missed[:5],
        "missed_black_blobs": len(missed_black),
        "missed_black_area_px": int(sum(item["area"] for item in missed_black)),
        "missed_black_top": missed_black[:8],
        "black_shadow_like": black_shadow_like[:10],
        "all_median_component_area": int(np.median(stats_all[1:, cv2.CC_STAT_AREA])) if n_all > 1 else 0,
        "black_median_component_area": int(np.median(stats_black[1:, cv2.CC_STAT_AREA])) if n_black > 1 else 0,
    }


def bbox_from_stats(stats: np.ndarray, label: int) -> list[int]:
    return [
        int(stats[label, cv2.CC_STAT_LEFT]),
        int(stats[label, cv2.CC_STAT_TOP]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
    ]


def connected_blob_summary(mask_bool: np.ndarray, min_area: int, light: np.ndarray, chroma: np.ndarray | None = None) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8) * 255, connectivity=8)
    out: list[dict] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        item = {
            "area": area,
            "bbox": bbox_from_stats(stats, label),
            "med_L": round(float(np.median(light[comp])), 1),
        }
        if chroma is not None:
            item["med_chroma"] = round(float(np.median(chroma[comp])), 1)
        out.append(item)
    out.sort(key=lambda item: -item["area"])
    return out


def image_output_name(input_root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(input_root).with_suffix("")
    return "_".join(rel.parts).replace("/", "_")


def display_name(name: str) -> str:
    return name.replace("Без Вспышка", "NoFlash").replace("Вспышка", "Flash")


def process_image(
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    do_shading: bool = True,
    target_short: int = 2400,
    logger: Logger = print,
) -> dict | None:
    raw = imread(image_path)
    if raw is None:
        logger(f"ERROR: cannot read {image_path}")
        return None
    stem = image_output_name(input_root, image_path)
    logger(f"[{stem}] {raw.shape[1]}x{raw.shape[0]}")

    paper_full, detect_info = detect_paper_mask(raw)
    crop_img, crop_paper, crop = crop_to_paper(raw, paper_full)
    detect_vis = draw_paper_detect(raw, paper_full, crop)

    if target_short > 0:
        crop_img = resize_short(crop_img, target_short, cv2.INTER_LANCZOS4)
        crop_paper = resize_short(crop_paper, target_short, cv2.INTER_NEAREST)
        crop_paper = ((crop_paper > 0).astype(np.uint8)) * 255

    normalized = normalize_scan(crop_img, crop_paper, do_shading=do_shading)
    normalized = white_balance_to_paper(normalized, crop_paper)

    all_mask, _, all_stats = segment_all_seeds(normalized, crop_paper)
    black_mask, _, black_stats = segment_black_seeds(normalized, all_mask, crop_paper)
    all_mask, black_mask, fixup_info = fixup_masks(normalized, crop_paper, all_mask, black_mask)

    # Refresh counts after fixup.
    all_contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_stats["seed_count"] = len(all_contours)
    all_stats["all_seed_area_px"] = int((all_mask > 0).sum())
    black_stats["black_seed_count"] = len(black_contours)
    black_stats["black_seed_area_px"] = int((black_mask > 0).sum())

    audit = audit_result(normalized, crop_paper, all_mask, black_mask)
    result = {
        "image": str(image_path),
        "name": stem,
        "input_size": [int(raw.shape[1]), int(raw.shape[0])],
        "crop": [int(v) for v in crop],
        "crop_size": [int(normalized.shape[1]), int(normalized.shape[0])],
        "target_short": int(target_short),
        "paper_detect": detect_info,
        "fixup": fixup_info,
        **all_stats,
        **black_stats,
        "white_seed_area_px": int(all_stats["all_seed_area_px"] - black_stats["black_seed_area_px"]),
        "black_to_all_ratio": audit["black_to_all_ratio"],
        "audit": audit,
    }

    imwrite(output_dir / f"{stem}_paper_detect.jpg", detect_vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
    imwrite(output_dir / f"{stem}_normalized.jpg", normalized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    imwrite(output_dir / f"{stem}_paper_mask.png", crop_paper)
    imwrite(output_dir / f"{stem}_all_mask.png", all_mask)
    imwrite(output_dir / f"{stem}_black_mask.png", black_mask)
    imwrite(output_dir / f"{stem}_all_overlay.jpg", visualize_all(normalized, all_mask, all_stats), [cv2.IMWRITE_JPEG_QUALITY, 92])
    imwrite(output_dir / f"{stem}_black_overlay.jpg", visualize_black(normalized, black_mask, black_stats), [cv2.IMWRITE_JPEG_QUALITY, 92])

    logger(
        f"  crop={crop} all={all_stats['seed_count']} area={all_stats['all_seed_area_px']} "
        f"black={black_stats['black_seed_count']} area={black_stats['black_seed_area_px']} "
        f"missed={audit['missed_seed_blobs']} missed_black={audit['missed_black_blobs']} "
        f"leaks={audit['leak_blobs']}"
    )
    return result


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("."))


def write_contact_sheet(output_dir: Path, rows: list[dict]) -> Path | None:
    tiles: list[np.ndarray] = []
    labels: list[str] = []
    for row in rows:
        stem = row["name"]
        all_path = output_dir / f"{stem}_all_overlay.jpg"
        black_path = output_dir / f"{stem}_black_overlay.jpg"
        all_img = imread(all_path)
        black_img = imread(black_path)
        if all_img is None or black_img is None:
            continue
        tile_h = 520
        all_resized = cv2.resize(all_img, (int(all_img.shape[1] * tile_h / all_img.shape[0]), tile_h), interpolation=cv2.INTER_AREA)
        black_resized = cv2.resize(black_img, (int(black_img.shape[1] * tile_h / black_img.shape[0]), tile_h), interpolation=cv2.INTER_AREA)
        tile = np.hstack([all_resized, black_resized])
        tiles.append(tile)
        labels.append(display_name(stem))
    if not tiles:
        return None
    target_w = max(tile.shape[1] for tile in tiles)
    padded: list[np.ndarray] = []
    for tile, label in zip(tiles, labels):
        if tile.shape[1] < target_w:
            pad = np.full((tile.shape[0], target_w - tile.shape[1], 3), 245, dtype=np.uint8)
            tile = np.hstack([tile, pad])
        label_bar = np.full((42, target_w, 3), 30, dtype=np.uint8)
        cv2.putText(label_bar, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        padded.append(np.vstack([label_bar, tile]))
    sheet = np.vstack(padded)
    path = output_dir / "contact_sheet.jpg"
    imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    recursive: bool = True,
    do_shading: bool = True,
    target_short: int = 2400,
    logger: Logger = print,
) -> int:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()

    if not input_path.exists():
        logger(f"ERROR: input path not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_path.parent if input_path.is_file() else input_path
    images = collect_images(input_path, recursive=recursive)
    if not images:
        logger("No input images found")
        return 1

    logger(f"Found {len(images)} images -> {output_dir}")
    results: list[dict] = []
    for image in images:
        row = process_image(
            image,
            input_root,
            output_dir,
            do_shading=do_shading,
            target_short=target_short,
            logger=logger,
        )
        if row is not None:
            results.append(row)

    summary_path = output_dir / "summary.json"
    audit_path = output_dir / "audit.json"
    summary_min = []
    audit_rows = []
    for row in results:
        audit = {"image": row["name"], "crop": row["crop"], **row["audit"]}
        audit_rows.append(audit)
        summary_min.append({
            "image": row["name"],
            "crop": row["crop"],
            "all_count": row["seed_count"],
            "all_area_px": row["all_seed_area_px"],
            "black_count": row["black_seed_count"],
            "black_area_px": row["black_seed_area_px"],
            "black_to_all_ratio": row["black_to_all_ratio"],
            "missed_seed_blobs": row["audit"]["missed_seed_blobs"],
            "missed_black_blobs": row["audit"]["missed_black_blobs"],
            "leak_blobs": row["audit"]["leak_blobs"],
            "black_shadow_like": len(row["audit"]["black_shadow_like"]),
        })
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet_path = write_contact_sheet(output_dir, results)

    logger("\nSUMMARY")
    for row in summary_min:
        logger(
            f"  {row['image']:18s} all={row['all_count']:4d} black={row['black_count']:4d} "
            f"missed={row['missed_seed_blobs']:2d} missed_black={row['missed_black_blobs']:2d} "
            f"leaks={row['leak_blobs']:2d} shadow_like={row['black_shadow_like']:2d}"
        )
    logger(f"\nsummary: {summary_path}")
    logger(f"audit:   {audit_path}")
    if sheet_path is not None:
        logger(f"sheet:   {sheet_path}")
    return 0


def launch_gui(
    initial_input: str = "",
    initial_output: str = "",
    initial_no_recursive: bool = False,
    initial_no_shading: bool = False,
    initial_target_short: int = 2400,
) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print(f"ERROR: tkinter is not available: {exc}")
        return 1

    class PipelineGUI:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("SeedDetect Pipeline v2")
            self.root.geometry("980x700")
            self.root.minsize(780, 540)

            self.log_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
            self.worker: threading.Thread | None = None

            self.input_var = tk.StringVar(value=initial_input)
            self.output_var = tk.StringVar(value=initial_output)
            self.no_recursive_var = tk.BooleanVar(value=initial_no_recursive)
            self.no_shading_var = tk.BooleanVar(value=initial_no_shading)
            self.target_short_var = tk.StringVar(value=str(initial_target_short))
            self.status_var = tk.StringVar(value="Готово")

            self._build_ui(ttk)
            self.root.after(100, self._poll_queue)

        def _build_ui(self, ttk_module) -> None:
            root_frame = ttk_module.Frame(self.root, padding=12)
            root_frame.pack(fill="both", expand=True)

            cfg_frame = ttk_module.LabelFrame(root_frame, text="Параметры", padding=10)
            cfg_frame.pack(fill="x")

            cfg_frame.columnconfigure(1, weight=1)

            ttk_module.Label(cfg_frame, text="Input:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk_module.Entry(cfg_frame, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", pady=4)
            ttk_module.Button(cfg_frame, text="Выбрать", command=self._pick_input).grid(row=0, column=2, padx=(8, 0), pady=4)

            ttk_module.Label(cfg_frame, text="Output:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk_module.Entry(cfg_frame, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=4)
            ttk_module.Button(cfg_frame, text="Выбрать", command=self._pick_output).grid(row=1, column=2, padx=(8, 0), pady=4)

            ttk_module.Label(cfg_frame, text="target-short:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk_module.Entry(cfg_frame, textvariable=self.target_short_var, width=12).grid(row=2, column=1, sticky="w", pady=4)

            checks = ttk_module.Frame(cfg_frame)
            checks.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 2))
            ttk_module.Checkbutton(checks, text="--no-recursive", variable=self.no_recursive_var).pack(side="left", padx=(0, 16))
            ttk_module.Checkbutton(checks, text="--no-shading", variable=self.no_shading_var).pack(side="left")

            status_frame = ttk_module.Frame(root_frame)
            status_frame.pack(fill="x", pady=(10, 0))
            ttk_module.Label(status_frame, text="Статус:").pack(side="left")
            ttk_module.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=(6, 0))

            log_frame = ttk_module.LabelFrame(root_frame, text="Лог", padding=8)
            log_frame.pack(fill="both", expand=True, pady=(10, 10))
            log_frame.rowconfigure(0, weight=1)
            log_frame.columnconfigure(0, weight=1)

            self.log_text = tk.Text(log_frame, wrap="word", state="disabled")
            self.log_text.grid(row=0, column=0, sticky="nsew")
            scroll = ttk_module.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
            scroll.grid(row=0, column=1, sticky="ns")
            self.log_text.configure(yscrollcommand=scroll.set)

            bottom_frame = ttk_module.Frame(root_frame)
            bottom_frame.pack(fill="x")
            self.run_button = ttk_module.Button(bottom_frame, text="Запустить pipeline", command=self._run_clicked)
            self.run_button.pack(side="bottom", fill="x")

        def _append_log(self, line: str) -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{line}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        def _pick_input(self) -> None:
            path = filedialog.askopenfilename(title="Выберите файл входа")
            if not path:
                path = filedialog.askdirectory(title="Или выберите папку входа")
            if path:
                self.input_var.set(path)

        def _pick_output(self) -> None:
            path = filedialog.askdirectory(title="Выберите папку вывода")
            if path:
                self.output_var.set(path)

        def _validate_inputs(self) -> dict | None:
            input_str = self.input_var.get().strip()
            output_str = self.output_var.get().strip()
            target_str = self.target_short_var.get().strip()

            if not input_str:
                messagebox.showerror("Ошибка", "Укажите путь input")
                return None
            if not output_str:
                messagebox.showerror("Ошибка", "Укажите путь output")
                return None

            input_path = Path(input_str).expanduser()
            if not input_path.exists():
                messagebox.showerror("Ошибка", f"Input не существует:\n{input_path}")
                return None

            try:
                target_short = int(target_str)
            except ValueError:
                messagebox.showerror("Ошибка", "target-short должен быть целым числом")
                return None

            if target_short <= 0:
                messagebox.showerror("Ошибка", "target-short должен быть положительным")
                return None

            return {
                "input_path": input_path,
                "output_path": Path(output_str).expanduser(),
                "recursive": not self.no_recursive_var.get(),
                "do_shading": not self.no_shading_var.get(),
                "target_short": target_short,
            }

        def _run_clicked(self) -> None:
            payload = self._validate_inputs()
            if payload is None:
                return
            if self.worker is not None and self.worker.is_alive():
                return

            self._append_log("-" * 72)
            self._append_log("Запуск pipeline")
            self.status_var.set("Выполняется...")
            self.run_button.configure(state="disabled")

            self.worker = threading.Thread(target=self._worker_main, args=(payload,), daemon=True)
            self.worker.start()

        def _worker_main(self, payload: dict) -> None:
            def log(message: str) -> None:
                self.log_queue.put(("log", message))

            try:
                rc = run_pipeline(
                    payload["input_path"],
                    payload["output_path"],
                    recursive=payload["recursive"],
                    do_shading=payload["do_shading"],
                    target_short=payload["target_short"],
                    logger=log,
                )
            except Exception:
                rc = 1
                log("ERROR: unhandled exception during run")
                log(traceback.format_exc().rstrip())
            self.log_queue.put(("done", rc))

        def _poll_queue(self) -> None:
            while True:
                try:
                    kind, payload = self.log_queue.get_nowait()
                except queue.Empty:
                    break

                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    rc = int(payload)
                    if rc == 0:
                        self.status_var.set("Готово")
                    else:
                        self.status_var.set("Ошибка")
                    self.run_button.configure(state="normal")

            self.root.after(100, self._poll_queue)

    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-masked seed segmentation pipeline v2")
    parser.add_argument("--input", help="Input image file or directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--no-recursive", action="store_true", help="Only process direct children of input directory")
    parser.add_argument("--no-shading", action="store_true", help="Disable flat-field shading correction")
    parser.add_argument("--target-short", type=int, default=2400, help="Upscale cropped paper image to this short side before segmentation")
    parser.add_argument("--gui", action="store_true", help="Launch tkinter GUI")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui:
        return launch_gui(
            initial_input=args.input or "",
            initial_output=args.output or "",
            initial_no_recursive=args.no_recursive,
            initial_no_shading=args.no_shading,
            initial_target_short=args.target_short,
        )

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --gui is used")

    return run_pipeline(
        Path(args.input),
        Path(args.output),
        recursive=not args.no_recursive,
        do_shading=not args.no_shading,
        target_short=args.target_short,
        logger=print,
    )


if __name__ == "__main__":
    raise SystemExit(main())
