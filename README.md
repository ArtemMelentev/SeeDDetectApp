# SeeDDetect Android v1

Offline Android prototype with local Python execution for seed segmentation.

## Stack

- Flutter UI
- Kotlin Android bridge
- Chaquopy for Python runtime
- OpenCV + NumPy for image processing

## Current flow (v1)

1. User picks one image from gallery or file picker.
2. Flutter sends URI to Android platform channel.
3. Kotlin copies file into app-internal cache run directory.
4. Chaquopy executes `segment_seeds_scan.analyze_image(...)`.
5. UI renders metrics and generated overlays/masks.

## Running on Windows (Android emulator)

Step-by-step checklist: `.kilo/plans/1776234800479-brave-planet.md`.

Notes:
- Current Chaquopy setup uses Python 3.10 on the build machine (`py -3.10 --version`).
- ABI filters are currently set to `x86_64` for emulator-focused runs.

## Python entrypoint contract

`analyze_image(input_path, output_dir) -> dict`

Response payload includes:

- Metrics: `seed_count`, `all_seed_area_px`, `black_seed_count`, `black_seed_area_px`
- Service fields: `image_w`, `image_h`, `processing_ms`
- Artifacts: absolute paths to all mask/overlay files
- Error payload format: `{ "ok": false, "error": { "code", "message", "details?" } }`

## Safety for large images

The pipeline applies a safe pre-resize when image dimensions exceed configured limits (`max_pixels`, `max_side`) to reduce OOM risk on mobile devices.

## Notes about iOS

This implementation is Android-only. iOS support should be added in a separate stage by moving processing core to cross-platform C++/OpenCV or by introducing a server mode (strategic decision pending).
