#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


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


def _score_paper_component(comp: np.ndarray, h: int, w: int) -> tuple[float, dict]:
    area = int((comp > 0).sum())
    img_area = float(h * w)
    if area <= 0:
        return -1.0, {"area": 0}
    frac = area / img_area
    x, y, bw, bh = component_bbox(comp)
    bbox_area = max(1, bw * bh)
    fill = area / float(bbox_area)
    touches = int(x <= 2) + int(y <= 2) + int(x + bw >= w - 2) + int(y + bh >= h - 2)
    score = 0.0
    # Strong preference for large but not whole-frame components.
    if 0.20 <= frac <= 0.92:
        score += 0.6 + 0.4 * (1.0 - abs(0.6 - frac))
    else:
        score -= 0.5
    score += 0.25 * fill
    if touches >= 4:
        score -= 0.8
    elif touches == 3:
        score -= 0.25
    info = {
        "area_frac_raw": round(frac, 4),
        "bbox_fill_raw": round(fill, 4),
        "bbox_raw": [x, y, bw, bh],
        "border_touches_raw": touches,
    }
    return score, info


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return 4 points in order: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Find a 4-point polygon approximation of contour, or None."""
    peri = cv2.arcLength(contour, True)
    for eps_frac in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _line_from_points(pts: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a line ax+by+c=0 (||(a,b)||=1) to Nx2 points via cv2.fitLine.

    Returns None if too few points.
    """
    if pts.shape[0] < 8:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    # Normal to direction (vx,vy) is (-vy, vx). Line: -vy*(x-x0) + vx*(y-y0) = 0
    a = -float(vy)
    b = float(vx)
    c = -(a * float(x0) + b * float(y0))
    n = float(np.hypot(a, b))
    if n < 1e-9:
        return None
    return a / n, b / n, c / n


def _intersect_lines(l1: tuple[float, float, float], l2: tuple[float, float, float]) -> np.ndarray | None:
    """Intersect two lines a*x + b*y + c = 0."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    # Solve [[a1,b1],[a2,b2]] @ [x,y] = [-c1,-c2]
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)


def _quad_from_paper_edges(comp_mask: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Hybrid quadrilateral fit: minAreaRect base + per-side line refit.

    Algorithm:
      1. Convex hull of the paper component (skips concavities from bridges
         into the wood table).
      2. Densify hull edges into a dense point cloud and tag each point as
         on-frame (lies on image border, i.e. paper runs off the photo) or
         off-frame (a real paper edge).
      3. Base quad = minAreaRect(hull). Its 4 sides give base line equations
         and a stable orientation, even when one side is fully cut off.
      4. For each of the 4 sides, assign off-frame hull points to that side
         (nearest-line + half-plane + orientation constraints) and refit
         the line if there are enough such points; otherwise keep the base
         line. K-means-like: 4 iterations.
      5. Intersect adjacent (refined) lines to produce the 4 corners.
    """
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, "no_contour"
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 80:
        return None, "tiny_contour"
    hull = cv2.convexHull(contour)
    if hull.shape[0] < 4:
        return None, "tiny_hull"
    H, W = comp_mask.shape
    border_eps = 3
    all_pts: list[np.ndarray] = []
    off_pts: list[np.ndarray] = []
    for i in range(hull.shape[0]):
        a = hull[i, 0].astype(np.float32)
        b = hull[(i + 1) % hull.shape[0], 0].astype(np.float32)
        seg_len = float(np.linalg.norm(b - a))
        n_steps = max(2, int(seg_len / 3.0))
        ts = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        seg = a[None, :] * (1.0 - ts[:, None]) + b[None, :] * ts[:, None]
        on_frame = (
            (seg[:, 0] <= border_eps)
            | (seg[:, 0] >= W - 1 - border_eps)
            | (seg[:, 1] <= border_eps)
            | (seg[:, 1] >= H - 1 - border_eps)
        )
        all_pts.append(seg)
        seg_off = seg[~on_frame]
        if seg_off.shape[0] > 0:
            off_pts.append(seg_off)
    if not all_pts:
        return None, "no_hull_pts"
    pts_all = np.concatenate(all_pts, axis=0)
    pts_off = np.concatenate(off_pts, axis=0) if off_pts else np.zeros((0, 2), dtype=np.float32)

    # Base = minAreaRect of the hull (stable; works even with one side cut).
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = _order_quad(box)  # TL, TR, BR, BL
    tl, tr, br, bl = box[0], box[1], box[2], box[3]

    def _line_through(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
        # Line from two points: normal = perpendicular to (q - p).
        dx = float(q[0] - p[0])
        dy = float(q[1] - p[1])
        a = -dy
        b = dx
        c = -(a * float(p[0]) + b * float(p[1]))
        n = float(np.hypot(a, b))
        if n < 1e-9:
            return 1.0, 0.0, -float(p[0])
        return a / n, b / n, c / n

    base_lines = [
        _line_through(tl, tr),  # 0 top
        _line_through(tr, br),  # 1 right
        _line_through(br, bl),  # 2 bottom
        _line_through(bl, tl),  # 3 left
    ]

    bx, by, bw, bh = cv2.boundingRect(hull)
    if bw < 40 or bh < 40:
        return None, "thin_bbox"

    def _signed_dist(line: tuple[float, float, float], p: np.ndarray) -> np.ndarray:
        a, b, c = line
        return a * p[:, 0] + b * p[:, 1] + c

    def _is_horizontal(line: tuple[float, float, float]) -> bool:
        a, b, _ = line
        return abs(b) > abs(a)

    # Per-side refit: only refit a side if enough off-frame points fall on it.
    min_side_pts = 25
    lines = list(base_lines)
    refits: list[bool] = [False, False, False, False]
    if pts_off.shape[0] >= 40:
        cx_box, cy_box = box.mean(axis=0)
        for _ in range(4):
            dists = np.stack([np.abs(_signed_dist(line, pts_off)) for line in lines], axis=1)
            cy = pts_off[:, 1]
            cx = pts_off[:, 0]
            big = 1e9
            masked = dists.copy()
            masked[cy > cy_box, 0] = big
            masked[cx < cx_box, 1] = big
            masked[cy < cy_box, 2] = big
            masked[cx > cx_box, 3] = big
            labels = np.argmin(masked, axis=1)

            new_lines: list[tuple[float, float, float]] = []
            new_refits: list[bool] = []
            for k in range(4):
                sel = pts_off[labels == k]
                if sel.shape[0] < min_side_pts:
                    new_lines.append(base_lines[k])
                    new_refits.append(False)
                    continue
                # Drop outliers (>3*MAD from base line) before refit.
                base_d = np.abs(_signed_dist(base_lines[k], sel))
                med = float(np.median(base_d))
                mad = float(np.median(np.abs(base_d - med))) + 1e-6
                inliers = sel[base_d < med + 6.0 * mad]
                if inliers.shape[0] < min_side_pts:
                    new_lines.append(base_lines[k])
                    new_refits.append(False)
                    continue
                line = _line_from_points(inliers)
                if line is None:
                    new_lines.append(base_lines[k])
                    new_refits.append(False)
                    continue
                # Orientation must match the base side.
                if _is_horizontal(line) != _is_horizontal(base_lines[k]):
                    new_lines.append(base_lines[k])
                    new_refits.append(False)
                    continue
                # Angle deviation from base must be small (<= ~12 deg).
                a0, b0, _ = base_lines[k]
                a1, b1, _ = line
                cos_ab = abs(a0 * a1 + b0 * b1)
                if cos_ab < 0.978:  # ~12 deg
                    new_lines.append(base_lines[k])
                    new_refits.append(False)
                    continue
                new_lines.append(line)
                new_refits.append(True)
            if all(np.allclose(np.array(a), np.array(b), atol=1e-3) for a, b in zip(lines, new_lines)):
                lines = new_lines
                refits = new_refits
                break
            lines = new_lines
            refits = new_refits

    line_top, line_right, line_bot, line_left = lines
    if not (_is_horizontal(line_top) and _is_horizontal(line_bot)):
        return None, "non_horizontal_top_bot"
    if _is_horizontal(line_left) or _is_horizontal(line_right):
        return None, "non_vertical_left_right"

    p_tl = _intersect_lines(line_top, line_left)
    p_tr = _intersect_lines(line_top, line_right)
    p_br = _intersect_lines(line_bot, line_right)
    p_bl = _intersect_lines(line_bot, line_left)
    if any(p is None for p in (p_tl, p_tr, p_br, p_bl)):
        return None, "parallel_edges"

    quad = np.stack([p_tl, p_tr, p_br, p_bl], axis=0).astype(np.float32)
    quad = _order_quad(quad)

    # Sanity vs hull (should not deviate wildly).
    hull_area = float(abs(cv2.contourArea(hull)))
    quad_area = float(abs(cv2.contourArea(quad)))
    if hull_area <= 0 or quad_area <= 0:
        return None, "zero_area"
    ratio = quad_area / hull_area
    if ratio < 0.85 or ratio > 1.25:
        return None, f"area_ratio_{ratio:.2f}"

    cx_h, cy_h = hull.reshape(-1, 2).mean(axis=0)
    cx_q, cy_q = quad.mean(axis=0)
    bbox_diag = float(np.hypot(bw, bh))
    if float(np.hypot(cx_q - cx_h, cy_q - cy_h)) > 0.12 * bbox_diag:
        return None, "centroid_far"

    n_refit = sum(1 for r in refits if r)
    status = f"hybrid_refit{n_refit}of4"
    return quad, status


def _moving_average_1d(vals: np.ndarray, k: int) -> np.ndarray:
    if vals.size == 0 or k <= 1:
        return vals
    k = min(k, vals.size)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(vals.astype(np.float32), kernel, mode="same")


def _first_paper_run(paper_like: np.ndarray, max_scan: int) -> int:
    scan = paper_like[:max_scan].astype(np.float32)
    if scan.size == 0:
        return 0
    run = max(4, min(14, int(round(scan.size * 0.05))))
    if scan.size < run:
        return 0
    support = _moving_average_1d(scan, run)
    hits = np.where(support >= 0.72)[0]
    if hits.size == 0:
        return 0
    trim = max(0, int(hits[0] - run // 2))
    return 0 if trim <= 2 else trim


def _refine_quad_by_border_color(img_bgr: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, dict]:
    """Trim table-coloured margins from an already-detected paper quad.

    The cyan field detector gives a good orientation, but a few pixels of
    saturated table can remain inside the quad. Work in the rectified quad so
    each side is a simple 1D colour-profile problem, then map the inset rect
    back through the inverse perspective transform.
    """
    quad = _order_quad(quad)
    tl, tr, br, bl = quad
    out_w = int(max(8.0, round(max(float(np.linalg.norm(br - bl)), float(np.linalg.norm(tr - tl))))))
    out_h = int(max(8.0, round(max(float(np.linalg.norm(tr - br)), float(np.linalg.norm(tl - bl))))))
    if out_w < 40 or out_h < 40:
        return quad, {"border_refine": "too_small"}

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    sat = hsv[:, :, 1].astype(np.float32)

    cy0, cy1 = int(out_h * 0.25), int(out_h * 0.75)
    cx0, cx1 = int(out_w * 0.25), int(out_w * 0.75)
    center = (slice(cy0, max(cy0 + 1, cy1)), slice(cx0, max(cx0 + 1, cx1)))
    ref_l = float(np.percentile(light[center], 75))
    ref_s = float(np.percentile(sat[center], 40))
    ref_c = float(np.percentile(chroma[center], 40))
    l_min = max(100.0, min(135.0, ref_l - 55.0))
    s_max = max(55.0, min(92.0, ref_s + 45.0))
    c_max = max(26.0, min(48.0, ref_c + 28.0))

    # Use the central cross-section for each side so side-table leakage does
    # not bias top/bottom scans, and vice versa.
    xs0, xs1 = int(out_w * 0.08), int(out_w * 0.92)
    ys0, ys1 = int(out_h * 0.08), int(out_h * 0.92)
    xs = slice(xs0, max(xs0 + 1, xs1))
    ys = slice(ys0, max(ys0 + 1, ys1))

    row_l = np.median(light[:, xs], axis=1)
    row_s = np.median(sat[:, xs], axis=1)
    row_c = np.median(chroma[:, xs], axis=1)
    col_l = np.median(light[ys, :], axis=0)
    col_s = np.median(sat[ys, :], axis=0)
    col_c = np.median(chroma[ys, :], axis=0)

    smooth_k = max(3, min(15, odd(min(out_w, out_h) * 0.004, 3)))
    row_l = _moving_average_1d(row_l, smooth_k)
    row_s = _moving_average_1d(row_s, smooth_k)
    row_c = _moving_average_1d(row_c, smooth_k)
    col_l = _moving_average_1d(col_l, smooth_k)
    col_s = _moving_average_1d(col_s, smooth_k)
    col_c = _moving_average_1d(col_c, smooth_k)

    row_paper = (row_l >= l_min) & (row_s <= s_max) & (row_c <= c_max)
    col_paper = (col_l >= l_min) & (col_s <= s_max) & (col_c <= c_max)
    max_y = max(6, int(round(out_h * 0.16)))
    max_x = max(6, int(round(out_w * 0.16)))
    top = _first_paper_run(row_paper, max_y)
    bottom = _first_paper_run(row_paper[::-1], max_y)
    left = _first_paper_run(col_paper, max_x)
    right = _first_paper_run(col_paper[::-1], max_x)

    if left + right > out_w * 0.28 or top + bottom > out_h * 0.28:
        return quad, {
            "border_refine": "rejected_large_trim",
            "border_trims_rect": [int(left), int(top), int(right), int(bottom)],
        }
    if max(left, top, right, bottom) <= 0:
        return quad, {
            "border_refine": "none",
            "border_thresholds": [round(l_min, 1), round(s_max, 1), round(c_max, 1)],
        }

    x0 = float(left)
    y0 = float(top)
    x1 = float(out_w - 1 - right)
    y1 = float(out_h - 1 - bottom)
    if x1 - x0 < out_w * 0.68 or y1 - y0 < out_h * 0.68:
        return quad, {
            "border_refine": "rejected_small_rect",
            "border_trims_rect": [int(left), int(top), int(right), int(bottom)],
        }

    Minv = cv2.getPerspectiveTransform(dst, quad.astype(np.float32))
    inset = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
    refined = cv2.perspectiveTransform(inset, Minv)[0]
    return _order_quad(refined), {
        "border_refine": "applied",
        "border_trims_rect": [int(left), int(top), int(right), int(bottom)],
        "border_thresholds": [round(l_min, 1), round(s_max, 1), round(c_max, 1)],
    }


def _cyan_field_mask(img_work: np.ndarray) -> tuple[np.ndarray, dict]:
    """Segment the light-cyan paper field directly via LAB colour.

    The cyan sheet is bright (L>140), cool (b<-2) and slightly green-cyan
    (a<+2). Wood table: dark, warm. Beige backing paper: bright but warm
    (b>0). White/grey miscellaneous paper scraps: b near zero.

    Pre-process with bilateral on a downscale to suppress seed/wood texture
    so the cyan sheet remains a single connected component after morph.
    """
    h, w = img_work.shape[:2]
    pp_short = 600
    short = min(h, w)
    pp_scale = 1.0 if short <= pp_short else pp_short / float(short)
    if pp_scale < 1.0:
        small = cv2.resize(img_work, (int(round(w * pp_scale)), int(round(h * pp_scale))), interpolation=cv2.INTER_AREA)
    else:
        small = img_work
    smooth = cv2.bilateralFilter(small, d=7, sigmaColor=35, sigmaSpace=9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    sh, sw = L.shape

    # Cyan sheet vs beige backing: bright (L>150), nearly neutral or slightly
    # green (a<2), and noticeably less yellow than beige (b<12). Beige
    # backing has a around +5..+10 and b around +15..+25.
    cyan_raw = ((L > 150.0) & (a < 2.0) & (b < 12.0)).astype(np.uint8) * 255
    raw_frac = float((cyan_raw > 0).sum()) / float(sh * sw)
    info = {"cyan_raw_frac": round(raw_frac, 4)}
    if raw_frac < 0.02:
        return np.zeros((h, w), dtype=np.uint8), info

    diag_s = float(np.hypot(sh, sw))
    open_k = odd(diag_s * 0.006, 5)
    close_seed_k = odd(diag_s * 0.020, 9)   # bridge across seed gaps inside the sheet
    close_final_k = odd(diag_s * 0.060, 21)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_seed_k, close_seed_k))
    close_final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_final_k, close_final_k))

    # Pass 1: gentle close to bridge cyan around individual seeds, then keep
    # the connected component that contains the image centre. This isolates
    # the cyan sheet from any unrelated cool-bright blobs (white paper
    # scraps, glare on the table, etc.) before we do the aggressive close.
    field1 = cv2.morphologyEx(cyan_raw, cv2.MORPH_CLOSE, close_seed_kernel, iterations=1)
    field1 = cv2.morphologyEx(field1, cv2.MORPH_OPEN, open_kernel, iterations=1)
    n1, labels1, stats1, _ = cv2.connectedComponentsWithStats(field1, connectivity=8)
    sheet_only = np.zeros_like(field1)
    if n1 > 1:
        cy_img, cx_img = sh // 2, sw // 2
        center_label = int(labels1[cy_img, cx_img])
        if center_label > 0 and stats1[center_label, cv2.CC_STAT_AREA] > 0.02 * sh * sw:
            sheet_only = (labels1 == center_label).astype(np.uint8) * 255
        else:
            idx = 1 + int(np.argmax(stats1[1:, cv2.CC_STAT_AREA]))
            sheet_only = (labels1 == idx).astype(np.uint8) * 255

    # Pass 2: aggressive close on the isolated sheet only, to swallow seed
    # piles that cover the centre of the sheet.
    field = cv2.morphologyEx(sheet_only, cv2.MORPH_CLOSE, close_final_kernel, iterations=2)
    field = fill_holes(field)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(field, connectivity=8)
    comp_small = np.zeros_like(field)
    if n > 1:
        cy_img, cx_img = sh // 2, sw // 2
        center_label = int(labels[cy_img, cx_img])
        if center_label > 0 and stats[center_label, cv2.CC_STAT_AREA] > 0.02 * sh * sw:
            comp_small = (labels == center_label).astype(np.uint8) * 255
        else:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            comp_small = (labels == idx).astype(np.uint8) * 255

    # Seeds occupy a large share of the cyan field, so the connected
    # component above only covers the *visible* cyan pixels (the ring
    # around the seed pile). Replace it with its convex hull so the entire
    # cyan rectangle, including the seed-covered interior, is captured.
    if comp_small.any():
        contours, _ = cv2.findContours(comp_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            comp_small = np.zeros_like(comp_small)
            cv2.fillPoly(comp_small, [hull], 255)

    if pp_scale < 1.0:
        comp = cv2.resize(comp_small, (w, h), interpolation=cv2.INTER_NEAREST)
        comp = ((comp > 0).astype(np.uint8)) * 255
    else:
        comp = comp_small
    info["cyan_comp_frac"] = round(float((comp > 0).sum()) / float(h * w), 4)
    return comp, info


def _table_mask(img_work: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect the wooden table region around the paper sheet.

    Pre-processing: bilateral filter on a slightly downscaled copy smooths
    out wood grain and seed texture so a single LAB cluster captures the
    table cleanly, then we upsample the result.

    Strategy: sample a thin border ring on every side, take the *darkest
    quartile* of each side independently, and combine these into a global
    "table colour" estimate. This works whether the table fills the ring
    (most photos) or only a thin sliver of one side (paper-fills-frame).
    """
    h, w = img_work.shape[:2]
    # Pre-process: downscale + bilateral. Faster and more robust to noise.
    pp_short = 600
    short = min(h, w)
    pp_scale = 1.0 if short <= pp_short else pp_short / float(short)
    if pp_scale < 1.0:
        small = cv2.resize(img_work, (int(round(w * pp_scale)), int(round(h * pp_scale))), interpolation=cv2.INTER_AREA)
    else:
        small = img_work
    smooth = cv2.bilateralFilter(small, d=7, sigmaColor=35, sigmaSpace=9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    sh, sw = L.shape

    border_w = max(6, int(round(0.04 * min(sh, sw))))

    def _side_dark_pixels(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if side == "top":
            sl = (slice(0, border_w), slice(0, sw))
        elif side == "bot":
            sl = (slice(sh - border_w, sh), slice(0, sw))
        elif side == "left":
            sl = (slice(0, sh), slice(0, border_w))
        else:
            sl = (slice(0, sh), slice(sw - border_w, sw))
        L_s = L[sl].reshape(-1)
        a_s = a[sl].reshape(-1)
        b_s = b[sl].reshape(-1)
        thr = float(np.percentile(L_s, 25))
        keep = L_s <= thr + 2.0
        return L_s[keep], a_s[keep], b_s[keep]

    L_dark_parts: list[np.ndarray] = []
    a_dark_parts: list[np.ndarray] = []
    b_dark_parts: list[np.ndarray] = []
    side_meds: list[float] = []
    for side in ("top", "bot", "left", "right"):
        Ls, as_, bs_ = _side_dark_pixels(side)
        if Ls.size > 0:
            L_dark_parts.append(Ls)
            a_dark_parts.append(as_)
            b_dark_parts.append(bs_)
            side_meds.append(float(np.median(Ls)))

    info: dict = {"side_dark_L_meds": [round(v, 1) for v in side_meds]}
    if not L_dark_parts:
        return np.ones((h, w), dtype=np.uint8) * 255, info

    # Filter: only sides whose dark median is plausibly table (L<140) feed
    # the global table-colour estimate. This avoids using a paper-only side.
    table_sides = [(Ls, as_, bs_) for (Ls, as_, bs_), m in zip(zip(L_dark_parts, a_dark_parts, b_dark_parts), side_meds) if m < 140.0]
    if not table_sides:
        info["table_present"] = False
        return np.ones((h, w), dtype=np.uint8) * 255, info
    L_anchor = np.concatenate([t[0] for t in table_sides])
    a_anchor = np.concatenate([t[1] for t in table_sides])
    b_anchor = np.concatenate([t[2] for t in table_sides])
    L_med = float(np.median(L_anchor))
    a_med = float(np.median(a_anchor))
    b_med = float(np.median(b_anchor))
    L_mad = float(np.median(np.abs(L_anchor - L_med))) + 1e-6
    a_mad = float(np.median(np.abs(a_anchor - a_med))) + 1e-6
    b_mad = float(np.median(np.abs(b_anchor - b_med))) + 1e-6

    info.update({
        "table_present": True,
        "ring_L_med": round(L_med, 1),
        "ring_a_med": round(a_med, 2),
        "ring_b_med": round(b_med, 2),
        "ring_L_mad": round(L_mad, 2),
    })

    # Mahalanobis-like distance with per-axis MAD scaling. Wood has tight
    # spread; paper is far in at least one axis.
    dL = (L - L_med) / max(L_mad, 3.0)
    da = (a - a_med) / max(a_mad, 2.0)
    db = (b - b_med) / max(b_mad, 2.0)
    d_ring = np.sqrt(dL * dL + da * da + db * db)
    L_cutoff = L_med + max(20.0, 8.0 * L_mad)
    table = (d_ring < 5.0) & (L < L_cutoff)

    # Paper = NOT table.
    paper_raw = (~table).astype(np.uint8) * 255
    diag_s = float(np.hypot(sh, sw))
    open_k = odd(diag_s * 0.005, 5)
    close_k = odd(diag_s * 0.014, 11)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    paper = cv2.morphologyEx(paper_raw, cv2.MORPH_OPEN, open_kernel, iterations=1)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    paper = fill_holes(paper)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=8)
    comp_small = np.zeros_like(paper)
    if n > 1:
        cy_img, cx_img = sh // 2, sw // 2
        center_label = int(labels[cy_img, cx_img])
        if center_label > 0 and stats[center_label, cv2.CC_STAT_AREA] > 0.05 * sh * sw:
            comp_small = (labels == center_label).astype(np.uint8) * 255
        else:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            comp_small = (labels == idx).astype(np.uint8) * 255

    # Upsample back to img_work resolution.
    if pp_scale < 1.0:
        comp = cv2.resize(comp_small, (w, h), interpolation=cv2.INTER_NEAREST)
        comp = ((comp > 0).astype(np.uint8)) * 255
    else:
        comp = comp_small
    return comp, info


def detect_paper_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect the paper sheet (beige + any inner coloured field) and return
    its 4 corners + filled mask.

    The goal is to exclude the wooden table from downstream seed
    segmentation. Paper may contain multiple colour zones (beige rim, cyan
    interior); we don't try to separate them -- we segment the *table* and
    take its complement, so the entire sheet ends up inside the ROI.
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

    # Primary: segment the cyan/blue field directly. The seed paper is a
    # consistent light-cyan (Lab a<0, b<0, L>140); the wooden table is dark
    # warm, the beige backing paper is bright but warm (b>0). A direct LAB
    # filter for "bright + cool" isolates the cyan sheet cleanly even when
    # a beige backing surrounds it.
    cyan_comp, cyan_info = _cyan_field_mask(img_work)
    if int(cyan_comp.sum()) >= 0.10 * h * w * 255:
        best_comp = cyan_comp
        table_info = {"strategy": "cyan_field", **cyan_info}
    else:
        # Fallback: subtract the wooden table -- works on already-cropped
        # photos where the entire frame is already the cyan sheet.
        best_comp, tinfo = _table_mask(img_work)
        table_info = {"strategy": "table_subtract", **tinfo}
    if int(best_comp.sum()) < 0.10 * h * w * 255:
        # Could not isolate paper from table -- give up and use full frame.
        mask = np.ones((orig_h, orig_w), dtype=np.uint8) * 255
        quad = np.array([[0, 0], [orig_w - 1, 0], [orig_w - 1, orig_h - 1], [0, orig_h - 1]], dtype=np.float32)
        info = {"fallback": "full_frame", "quad": quad.tolist(), **table_info}
        return mask, info
    best_score, best_info = _score_paper_component(best_comp, h, w)
    best_info.update(table_info)
    close_k = odd(diag * 0.012, 11)

    # Find the paper contour and reduce it to 4 corners.
    contours, _ = cv2.findContours(best_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad_work: np.ndarray | None = None
    quad_source = "none"
    # Preferred: fit 4 lines to the boundary -- robust to small bridges
    # between the paper component and the wood table.
    quad_work, edge_status = _quad_from_paper_edges(best_comp)
    if quad_work is not None:
        quad_source = edge_status
    if contours and quad_work is None:
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        # Fallback 1: polyDP 4-gon.
        quad_work = _quad_from_contour(contour)
        if quad_work is not None:
            quad_area = float(cv2.contourArea(quad_work.astype(np.float32)))
            if quad_area <= 0 or contour_area / quad_area < 0.85 or contour_area / quad_area > 1.15:
                quad_work = None
            else:
                quad_source = f"approxPolyDP({edge_status})"
        if quad_work is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            quad_work = box.astype(np.float32)
            quad_source = f"minAreaRect({edge_status})"

    if quad_work is None:
        # Last-resort: use bbox.
        x, y, bw, bh = component_bbox(best_comp)
        quad_work = np.array(
            [[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]],
            dtype=np.float32,
        )
        quad_source = "bbox"

    quad_work = _order_quad(quad_work)
    quad_work, border_info = _refine_quad_by_border_color(img_work, quad_work)

    # Inward shrink along corner-vectors (keeps the warp safely inside the
    # paper, away from the wood edge and any glow/shadow band along it).
    # Kept small so seeds resting on the very edge of the sheet stay inside.
    cx, cy = quad_work.mean(axis=0)
    shrink = max(4.0, diag * 0.008)
    centered = quad_work - np.array([cx, cy])
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    quad_shrunk = quad_work - centered / norms * shrink

    # Build a clean polygon mask at work resolution from the shrunk quad.
    quad_mask_work = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(quad_mask_work, [quad_shrunk.astype(np.int32)], 255)

    # Scale the quad and mask back to original image coordinates.
    if scale < 1.0:
        quad_orig = quad_shrunk / scale
        paper_mask = cv2.resize(quad_mask_work, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        paper_mask = ((paper_mask > 0).astype(np.uint8)) * 255
    else:
        quad_orig = quad_shrunk
        paper_mask = quad_mask_work

    quad_orig = quad_orig.astype(np.float32)

    x, y, bw, bh = component_bbox(paper_mask)
    best_info.update({
        "area_frac": round(float((paper_mask > 0).sum()) / float(orig_h * orig_w), 4),
        "bbox": [x, y, bw, bh],
        "close_k": close_k,
        "work_scale": round(scale, 4),
        "score": round(float(best_score), 4),
        "quad_source": quad_source,
        "quad": quad_orig.tolist(),
    })
    best_info.update(border_info)
    return paper_mask, best_info


def draw_paper_detect(img_bgr: np.ndarray, paper_mask: np.ndarray, quad: np.ndarray) -> np.ndarray:
    vis = img_bgr.copy()
    overlay = vis.copy()
    overlay[paper_mask > 0] = (255, 180, 0)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [poly], isClosed=True, color=(0, 0, 255), thickness=3)
    for i, p in enumerate(quad.astype(int)):
        cv2.circle(vis, tuple(p), 10, (0, 255, 255), -1)
        cv2.putText(vis, "TL TR BR BL".split()[i], tuple(p + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def warp_to_paper(img_bgr: np.ndarray, paper_mask: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Perspective-warp the paper quad to an axis-aligned rectangle.

    The output image contains only the paper interior, so wood corners cannot
    leak into the seed segmentation.
    """
    quad = _order_quad(np.asarray(quad, dtype=np.float32))
    tl, tr, br, bl = quad
    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    out_w = int(max(8.0, round(max(width_a, width_b))))
    out_h = int(max(8.0, round(max(height_a, height_b))))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    warped_mask = cv2.warpPerspective(paper_mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = ((warped_mask > 0).astype(np.uint8)) * 255
    info = {
        "out_size": [out_w, out_h],
        "src_quad": quad.tolist(),
        "M": M.tolist(),
    }
    return warped, warped_mask, info


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
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        yy = slice(y, y + bh)
        xx = slice(x, x + bw)
        comp = labels[yy, xx] == label
        l_in = light[yy, xx][comp]
        med_l = float(np.median(l_in))
        mean_l = float(l_in.mean())
        std_l = float(l_in.std())
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
        out_roi = out[yy, xx]
        out_roi[comp] = 255
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
    paper_bool = paper_mask > 0
    paper_pix = paper_bool & (all_mask == 0) & (light > 190) & (chroma < 15)
    paper_l = float(np.median(light[paper_pix])) if int(np.count_nonzero(paper_pix)) > 1000 else 240.0

    holes = interior_holes(all_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    accepted = np.zeros_like(holes)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        yy = slice(y, y + bh)
        xx = slice(x, x + bw)
        comp = labels[yy, xx] == label
        if int(np.count_nonzero(paper_bool[yy, xx] & comp)) < area * 0.95:
            continue
        med_l = float(np.median(light[yy, xx][comp]))
        mean_chroma = float(chroma[yy, xx][comp].mean())
        if mean_chroma >= 45.0:
            continue
        accept = False
        if area < 200 and med_l < paper_l - 20.0:
            accept = True
        elif area < 2500 and med_l < paper_l - 45.0:
            accept = True
        elif med_l < paper_l - 80.0:
            accept = True
        if accept:
            accepted_roi = accepted[yy, xx]
            accepted_roi[comp] = 255
    fixed_all = cv2.bitwise_or(all_mask, accepted)
    fixed_all = cv2.bitwise_and(fixed_all, paper_mask)

    recovered = recover_black_pixels(img_bgr, fixed_all, paper_mask, light)
    fixed_black = cv2.bitwise_or(black_mask, recovered)
    fixed_black = cv2.bitwise_and(fixed_black, fixed_all)
    fixed_black = cv2.bitwise_and(fixed_black, paper_mask)
    info = {
        "paper_L": round(paper_l, 1),
        "all_added_px": int(cv2.countNonZero(fixed_all) - cv2.countNonZero(all_mask)),
        "black_added_px": int(cv2.countNonZero(fixed_black) - cv2.countNonZero(black_mask)),
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
    metrics_only: bool = False,
) -> dict | None:
    raw = imread(image_path)
    if raw is None:
        print(f"ERROR: cannot read {image_path}")
        return None
    stem = image_output_name(input_root, image_path)
    print(f"[{stem}] {raw.shape[1]}x{raw.shape[0]}")

    paper_full, detect_info = detect_paper_mask(raw)
    quad = np.array(detect_info["quad"], dtype=np.float32)
    crop_img, crop_paper, warp_info = warp_to_paper(raw, paper_full, quad)
    detect_info["warp"] = {"out_size": warp_info["out_size"]}
    crop = (0, 0, crop_img.shape[1], crop_img.shape[0])

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
    all_stats["all_seed_area_px"] = int(cv2.countNonZero(all_mask))
    black_stats["black_seed_count"] = len(black_contours)
    black_stats["black_seed_area_px"] = int(cv2.countNonZero(black_mask))

    ratio = round(float(black_stats["black_seed_area_px"]) / max(1, int(all_stats["all_seed_area_px"])), 4)
    if metrics_only:
        audit = {
            "size": [int(normalized.shape[1]), int(normalized.shape[0])],
            "all_count": int(all_stats["seed_count"]),
            "all_area_px": int(all_stats["all_seed_area_px"]),
            "black_count": int(black_stats["black_seed_count"]),
            "black_area_px": int(black_stats["black_seed_area_px"]),
            "black_to_all_ratio": ratio,
            "leak_blobs": 0,
            "leak_area_px": 0,
            "leak_top": [],
            "missed_seed_blobs": 0,
            "missed_seed_area_px": 0,
            "missed_seed_top": [],
            "missed_black_blobs": 0,
            "missed_black_area_px": 0,
            "missed_black_top": [],
            "black_shadow_like": [],
        }
    else:
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

    if not metrics_only:
        detect_vis = draw_paper_detect(raw, paper_full, quad)
        imwrite(output_dir / f"{stem}_paper_detect.jpg", detect_vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
        imwrite(output_dir / f"{stem}_normalized.jpg", normalized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        imwrite(output_dir / f"{stem}_paper_mask.png", crop_paper)
        imwrite(output_dir / f"{stem}_all_mask.png", all_mask)
        imwrite(output_dir / f"{stem}_black_mask.png", black_mask)
        imwrite(output_dir / f"{stem}_all_overlay.jpg", visualize_all(normalized, all_mask, all_stats), [cv2.IMWRITE_JPEG_QUALITY, 92])
        imwrite(output_dir / f"{stem}_black_overlay.jpg", visualize_black(normalized, black_mask, black_stats), [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(
        f"  crop={crop} all={all_stats['seed_count']} area={all_stats['all_seed_area_px']} "
        f"black={black_stats['black_seed_count']} area={black_stats['black_seed_area_px']} "
        f"ratio={audit['black_to_all_ratio']:.4f} "
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-masked seed segmentation pipeline v2")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-recursive", action="store_true", help="Only process direct children of input directory")
    parser.add_argument("--no-shading", action="store_true", help="Disable flat-field shading correction")
    parser.add_argument("--target-short", type=int, default=2400, help="Upscale cropped paper image to this short side before segmentation")
    parser.add_argument("--metrics-only", action="store_true", help="Only compute JSON area ratios; skip audit images, masks, overlays, and contact sheet")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_path.parent if input_path.is_file() else input_path
    images = collect_images(input_path, recursive=not args.no_recursive)
    if not images:
        raise SystemExit("No input images found")

    print(f"Found {len(images)} images -> {output_dir}")
    results: list[dict] = []
    for image in images:
        row = process_image(
            image,
            input_root,
            output_dir,
            do_shading=not args.no_shading,
            target_short=args.target_short,
            metrics_only=args.metrics_only,
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
            "missed_seed_blobs": row["audit"].get("missed_seed_blobs", 0),
            "missed_black_blobs": row["audit"].get("missed_black_blobs", 0),
            "leak_blobs": row["audit"].get("leak_blobs", 0),
            "black_shadow_like": len(row["audit"].get("black_shadow_like", [])),
        })
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet_path = None if args.metrics_only else write_contact_sheet(output_dir, results)

    print("\nSUMMARY")
    for row in summary_min:
        print(
            f"  {row['image']:18s} all={row['all_count']:4d} black={row['black_count']:4d} "
            f"missed={row['missed_seed_blobs']:2d} missed_black={row['missed_black_blobs']:2d} "
            f"leaks={row['leak_blobs']:2d} shadow_like={row['black_shadow_like']:2d}"
        )
    print(f"\nsummary: {summary_path}")
    print(f"audit:   {audit_path}")
    if sheet_path is not None:
        print(f"sheet:   {sheet_path}")


if __name__ == "__main__":
    main()
