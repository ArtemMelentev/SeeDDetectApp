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
- Chaquopy ABI filters are set to `x86_64` and `arm64-v8a` in `defaultConfig` to keep both emulator and phone support.

## Phone release runbook (arm64-v8a)

1. Create a local release keystore (example path: `android/app/upload-keystore.jks`).
2. Create `android/key.properties` (file is ignored by git):

```properties
storePassword=<your-store-password>
keyPassword=<your-key-password>
keyAlias=<your-key-alias>
storeFile=upload-keystore.jks
```

3. Build release APK for phone target:

```bash
flutter clean
flutter pub get
flutter build apk --release --target-platform android-arm64
```

4. Use the phone artifact:

`build/app/outputs/flutter-apk/app-release.apk`

5. Install on phone:

```bash
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

6. Emulator regression after release changes:

```bash
flutter run -d <your_x86_64_emulator_id>
```

Smoke-check on emulator and phone: pick file -> wait for processing -> verify metrics and artifacts are shown.

## Troubleshooting

- Chaquopy timeout from `chaquo.com`: retry on stable network or VPN; if needed, use a local wheel cache and pass pip `--find-links`.
- No wheel for selected ABI/Python/package pin: choose a compatible package version for current `chaquopy + python + abi` combination.

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
