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


# ──────────────────────────────────────────────────────────────────────
#  Pass 1 — all seeds (light + dark)
# ──────────────────────────────────────────────────────────────────────

def segment_all_seeds(img, edge_margin=10, dilation_k=51, diff_thresh=15):
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

def segment_black_seeds(img, all_mask=None, edge_margin=10):
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


def analyze_image(input_path, output_dir, max_pixels=DEFAULT_MAX_PIXELS, max_side=DEFAULT_MAX_SIDE):
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
            all_mask, all_cnt, all_stats = segment_all_seeds(prepared_img)
            blk_mask, blk_cnt, blk_stats = segment_black_seeds(prepared_img, all_mask)
        except cv2.error as exc:
            return _error_payload("opencv_error", "OpenCV processing failed", str(exc))

        all_mask_path = output_dir / f"{stem}_all_mask.png"
        all_overlay_path = output_dir / f"{stem}_all_overlay.jpg"
        black_mask_path = output_dir / f"{stem}_black_mask.png"
        black_overlay_path = output_dir / f"{stem}_black_overlay.jpg"

        if not _safe_cv2_write(all_mask_path, all_mask, ".png"):
            return _error_payload("write_error", f"Cannot write artifact: {all_mask_path}")
        try:
            visualize_all_seeds(prepared_img, all_mask, all_cnt, all_stats, str(all_overlay_path))
        except OSError as exc:
            return _error_payload("write_error", "Cannot write all-seed overlay", str(exc))

        if not _safe_cv2_write(black_mask_path, blk_mask, ".png"):
            return _error_payload("write_error", f"Cannot write artifact: {black_mask_path}")
        try:
            visualize_black_seeds(prepared_img, blk_mask, blk_cnt, blk_stats, str(black_overlay_path))
        except OSError as exc:
            return _error_payload("write_error", "Cannot write black-seed overlay", str(exc))

        processing_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "image": str(input_path),
            "image_w": int(image_w),
            "image_h": int(image_h),
            "processing_ms": processing_ms,
            **all_stats,
            **blk_stats,
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
