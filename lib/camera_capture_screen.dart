import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class CameraCaptureScreen extends StatefulWidget {
  const CameraCaptureScreen({super.key});

  @override
  State<CameraCaptureScreen> createState() => _CameraCaptureScreenState();
}

class _CameraCaptureScreenState extends State<CameraCaptureScreen> {
  static const _flashModes = <FlashMode>[
    FlashMode.off,
    FlashMode.auto,
    FlashMode.always,
    FlashMode.torch,
  ];

  CameraController? _controller;
  bool _isInitializing = true;
  bool _isCapturing = false;
  String? _errorMessage;
  FlashMode _flashMode = FlashMode.off;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  double _currentZoom = 1.0;
  double _baseZoom = 1.0;

  bool get _isReady => _controller != null && _controller!.value.isInitialized;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    setState(() {
      _isInitializing = true;
      _errorMessage = null;
    });

    try {
      final cameras = await availableCameras();
      if (!mounted) {
        return;
      }

      if (cameras.isEmpty) {
        setState(() {
          _isInitializing = false;
          _errorMessage = 'Камера недоступна на устройстве.';
        });
        return;
      }

      final selectedCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        selectedCamera,
        ResolutionPreset.max,
        enableAudio: false,
      );

      await controller.initialize();

      debugPrint(
        '[CameraCapture] preset=max '
        'camera=${controller.description.name} '
        'lens=${controller.description.lensDirection} '
        'sensor=${controller.description.sensorOrientation} '
        'previewSize=${controller.value.previewSize} '
        'aspectRatio=${controller.value.aspectRatio} '
        'streaming=${controller.value.isStreamingImages} '
        'recording=${controller.value.isRecordingVideo}',
      );

      final minZoom = await controller.getMinZoomLevel();
      final maxZoom = await controller.getMaxZoomLevel();

      try {
        await controller.setFlashMode(FlashMode.off);
      } on CameraException {}

      if (!mounted) {
        await controller.dispose();
        return;
      }

      final previous = _controller;
      setState(() {
        _controller = controller;
        _isInitializing = false;
        _errorMessage = null;
        _minZoom = minZoom;
        _maxZoom = maxZoom;
        _currentZoom = minZoom;
        _flashMode = FlashMode.off;
      });
      await previous?.dispose();
    } on CameraException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _isInitializing = false;
        _errorMessage = _localizedCameraMessage(error);
      });
    } catch (_) {
      if (!mounted) {
        return;
      }

      setState(() {
        _isInitializing = false;
        _errorMessage = 'Не удалось открыть камеру на устройстве.';
      });
    }
  }

  String _localizedCameraMessage(CameraException error) {
    switch (error.code) {
      case 'CameraAccessDenied':
      case 'CameraAccessDeniedWithoutPrompt':
      case 'CameraAccessRestricted':
        return 'Доступ к камере отклонен. Разрешите его в настройках устройства.';
      case 'CameraNotAvailable':
      case 'cameraUnavailable':
        return 'Камера недоступна на устройстве.';
      default:
        return 'Не удалось открыть камеру на устройстве.';
    }
  }

  Future<void> _cycleFlashMode() async {
    final controller = _controller;
    if (!_isReady || controller == null) {
      return;
    }

    final startIndex = _flashModes.indexOf(_flashMode);
    final index = startIndex >= 0 ? startIndex : 0;

    for (var step = 1; step <= _flashModes.length; step++) {
      final nextMode = _flashModes[(index + step) % _flashModes.length];
      try {
        await controller.setFlashMode(nextMode);
        if (!mounted) {
          return;
        }

        setState(() => _flashMode = nextMode);
        return;
      } on CameraException {
        continue;
      }
    }

    if (!mounted) {
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Управление вспышкой недоступно на устройстве.')),
    );
  }

  Future<void> _setZoomLevel(double value) async {
    final controller = _controller;
    if (!_isReady || controller == null) {
      return;
    }

    final target = value.clamp(_minZoom, _maxZoom).toDouble();
    try {
      await controller.setZoomLevel(target);
      if (!mounted) {
        return;
      }

      setState(() => _currentZoom = target);
    } on CameraException {
      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Управление зумом недоступно на устройстве.')),
      );
    }
  }

  void _handleScaleStart(ScaleStartDetails details) {
    _baseZoom = _currentZoom;
  }

  void _handleScaleUpdate(ScaleUpdateDetails details) {
    if (!_isReady) {
      return;
    }

    final target = (_baseZoom * details.scale).clamp(_minZoom, _maxZoom).toDouble();
    if ((target - _currentZoom).abs() < 0.02) {
      return;
    }

    _setZoomLevel(target);
  }

  Future<void> _capturePhoto() async {
    final controller = _controller;
    if (!_isReady || controller == null || _isCapturing) {
      return;
    }

    setState(() => _isCapturing = true);
    try {
      final file = await controller.takePicture();
      if (!mounted) {
        return;
      }

      Navigator.of(context).pop(file);
    } on CameraException catch (error) {
      if (!mounted) {
        return;
      }

      final message = switch (error.code) {
        'CameraAccessDenied' ||
        'CameraAccessDeniedWithoutPrompt' ||
        'CameraAccessRestricted' =>
          'Доступ к камере отклонен. Разрешите его в настройках устройства.',
        _ => 'Не удалось сделать снимок. Попробуйте еще раз.',
      };

      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
    } catch (_) {
      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Не удалось сделать снимок. Попробуйте еще раз.')),
      );
    } finally {
      if (mounted) {
        setState(() => _isCapturing = false);
      }
    }
  }

  IconData _flashIcon(FlashMode mode) {
    return switch (mode) {
      FlashMode.auto => Icons.flash_auto,
      FlashMode.always => Icons.flash_on,
      FlashMode.torch => Icons.highlight,
      _ => Icons.flash_off,
    };
  }

  String _flashLabel(FlashMode mode) {
    return switch (mode) {
      FlashMode.auto => 'Авто',
      FlashMode.always => 'Вкл',
      FlashMode.torch => 'Фонарь',
      _ => 'Выкл',
    };
  }

  @override
  Widget build(BuildContext context) {
    final canZoom = (_maxZoom - _minZoom) > 0.05;

    return Scaffold(
      appBar: AppBar(title: const Text('Сделать снимок')),
      backgroundColor: Colors.black,
      body: _isInitializing
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 12),
                  Text('Инициализация камеры...', style: TextStyle(color: Colors.white)),
                ],
              ),
            )
          : _errorMessage != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _errorMessage!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                    ),
                    const SizedBox(height: 16),
                    FilledButton.icon(
                      onPressed: () => Navigator.of(context).pop(),
                      icon: const Icon(Icons.close),
                      label: const Text('Закрыть'),
                    ),
                  ],
                ),
              ),
            )
          : !_isReady
          ? const SizedBox.shrink()
          : Stack(
              fit: StackFit.expand,
              children: [
                GestureDetector(
                  onScaleStart: canZoom ? _handleScaleStart : null,
                  onScaleUpdate: canZoom ? _handleScaleUpdate : null,
                  child: CameraPreview(_controller!),
                ),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: SafeArea(
                    top: false,
                    child: Container(
                      width: double.infinity,
                      color: Colors.black.withOpacity(0.45),
                      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Row(
                            children: [
                              const Icon(Icons.zoom_in, color: Colors.white),
                              Expanded(
                                child: Slider(
                                  value: _currentZoom.clamp(_minZoom, _maxZoom),
                                  min: _minZoom,
                                  max: _maxZoom,
                                  onChanged: canZoom
                                      ? (value) {
                                          _setZoomLevel(value);
                                        }
                                      : null,
                                ),
                              ),
                              Text(
                                '${_currentZoom.toStringAsFixed(1)}x',
                                style: const TextStyle(color: Colors.white),
                              ),
                            ],
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              FilledButton.tonalIcon(
                                onPressed: _cycleFlashMode,
                                icon: Icon(_flashIcon(_flashMode)),
                                label: Text(_flashLabel(_flashMode)),
                              ),
                              const SizedBox(width: 20),
                              FloatingActionButton.large(
                                heroTag: 'camera_capture_button',
                                onPressed: _isCapturing ? null : _capturePhoto,
                                backgroundColor: Colors.white,
                                foregroundColor: Colors.black,
                                child: _isCapturing
                                    ? const SizedBox(
                                        width: 26,
                                        height: 26,
                                        child: CircularProgressIndicator(strokeWidth: 2),
                                      )
                                    : const Icon(Icons.camera_alt, size: 30),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}
