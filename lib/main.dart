import 'dart:io';

import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'camera_capture_screen.dart';

const _channel = MethodChannel('seed_detect/analyzer');

const Map<String, String> _errorByCodeRu = {
  'pdf_image_not_found':
      'В PDF не найдено изображений. Выберите PDF с одним встроенным изображением.',
  'pdf_multiple_images':
      'В PDF найдено несколько изображений. Нужен PDF ровно с одним изображением.',
  'pdf_extract_failed':
      'Не удалось извлечь изображение из PDF. Проверьте файл и попробуйте снова.',
  'invalid_args': 'Переданы некорректные параметры для анализа.',
  'native_error': 'Ошибка Android-модуля при запуске анализа.',
};

void main() {
  runApp(const SeedDetectApp());
}

enum AnalyzeState { idle, picking, processing, done, error }

class SeedDetectApp extends StatelessWidget {
  const SeedDetectApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Seed Detect',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF145A32)),
        useMaterial3: true,
      ),
      home: const AnalyzeScreen(),
    );
  }
}

class AnalyzeScreen extends StatefulWidget {
  const AnalyzeScreen({super.key});

  @override
  State<AnalyzeScreen> createState() => _AnalyzeScreenState();
}

class _AnalyzeScreenState extends State<AnalyzeScreen>
    with WidgetsBindingObserver {
  final ImagePicker _imagePicker = ImagePicker();

  AnalyzeState _state = AnalyzeState.idle;
  Map<String, dynamic>? _result;
  String? _errorMessage;
  bool _isConsumingSharedInput = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _consumeSharedInputFromPlatform();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _consumeSharedInputFromPlatform();
    }
  }

  Future<void> _consumeSharedInputFromPlatform() async {
    if (_isConsumingSharedInput || !mounted) {
      return;
    }

    _isConsumingSharedInput = true;
    try {
      final raw = await _channel.invokeMethod<dynamic>('consumeSharedInput');
      final normalized = _normalize(raw);
      if (normalized is! Map<String, dynamic>) {
        return;
      }

      final sharedUri = normalized['inputUri']?.toString();
      if (sharedUri == null || sharedUri.isEmpty) {
        return;
      }

      final notice = _buildSharedImportNotice(normalized);
      if (mounted && notice != null && notice.isNotEmpty) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) {
            return;
          }

          final messenger = ScaffoldMessenger.maybeOf(context);
          messenger?.showSnackBar(SnackBar(content: Text(notice)));
        });
      }

      await _runAnalysis(sharedUri);
    } on MissingPluginException {
      return;
    } on PlatformException catch (exc) {
      if (!mounted) {
        return;
      }

      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = exc.message ?? 'Не удалось получить файл из шаринга';
      });
    } finally {
      _isConsumingSharedInput = false;
    }
  }

  String? _buildSharedImportNotice(Map<String, dynamic> payload) {
    final summary = payload['summaryMessage']?.toString();
    if (summary != null && summary.isNotEmpty) {
      return summary;
    }

    final totalCount = int.tryParse(payload['totalCount']?.toString() ?? '') ?? 0;
    final importedCount =
        int.tryParse(payload['importedCount']?.toString() ?? '') ?? 0;
    final failedCount = int.tryParse(payload['failedCount']?.toString() ?? '') ?? 0;

    if (totalCount <= 1) {
      return null;
    }

    if (failedCount > 0) {
      return 'Импортировано $importedCount из $totalCount файлов. Открыт первый поддерживаемый файл.';
    }

    return 'Импортировано $importedCount файлов. Открыт первый файл.';
  }

  Future<void> _pickFromGallery() async {
    setState(() {
      _state = AnalyzeState.picking;
      _errorMessage = null;
    });

    final file = await _imagePicker.pickImage(source: ImageSource.gallery);
    if (file == null) {
      setState(() => _state = AnalyzeState.idle);
      return;
    }

    final uri = Uri.file(file.path).toString();
    await _runAnalysis(uri);
  }

  Future<void> _pickFromFiles() async {
    setState(() {
      _state = AnalyzeState.picking;
      _errorMessage = null;
    });

    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'pdf'],
      withData: false,
    );

    final path = picked?.files.single.path;
    if (path == null || path.isEmpty) {
      setState(() => _state = AnalyzeState.idle);
      return;
    }

    final uri = Uri.file(path).toString();
    await _runAnalysis(uri);
  }

  Future<void> _captureFromCamera() async {
    setState(() {
      _state = AnalyzeState.picking;
      _errorMessage = null;
    });

    try {
      final photo = await Navigator.of(context).push<XFile>(
        MaterialPageRoute(builder: (_) => const CameraCaptureScreen()),
      );

      if (!mounted) {
        return;
      }

      if (photo == null) {
        setState(() => _state = AnalyzeState.idle);
        return;
      }

      final uri = Uri.file(photo.path).toString();
      await _runAnalysis(uri);
    } on CameraException catch (exc) {
      if (!mounted) {
        return;
      }

      final message = switch (exc.code) {
        'CameraAccessDenied' ||
        'CameraAccessDeniedWithoutPrompt' ||
        'CameraAccessRestricted' =>
          'Доступ к камере отклонен. Разрешите его в настройках устройства.',
        _ => 'Не удалось открыть камеру на устройстве.',
      };

      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = message;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }

      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = 'Не удалось открыть камеру на устройстве.';
      });
    }
  }

  Future<void> _runAnalysis(String inputUri) async {
    setState(() {
      _state = AnalyzeState.processing;
      _result = null;
      _errorMessage = null;
    });

    try {
      final raw = await _channel.invokeMethod<dynamic>('analyzeImage', {
        'inputUri': inputUri,
      });

      final normalized = _normalize(raw);
      if (normalized is! Map<String, dynamic>) {
        throw const FormatException('Native payload has invalid format');
      }

      final ok = normalized['ok'] == true;
      if (ok) {
        setState(() {
          _state = AnalyzeState.done;
          _result = normalized;
        });
      } else {
        final message = _localizedErrorMessage(normalized['error']);
        setState(() {
          _state = AnalyzeState.error;
          _result = normalized;
          _errorMessage = message;
        });
      }
    } on PlatformException catch (exc) {
      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = exc.message ?? 'Ошибка платформенного канала';
      });
    } catch (exc) {
      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = 'Не удалось выполнить анализ файла на устройстве';
      });
    }
  }

  @visibleForTesting
  Future<void> runAnalysisForTest(String inputUri) => _runAnalysis(inputUri);

  String _localizedErrorMessage(dynamic errorRaw) {
    if (errorRaw is Map<String, dynamic>) {
      final code = errorRaw['code']?.toString();
      if (code != null && _errorByCodeRu.containsKey(code)) {
        return _errorByCodeRu[code]!;
      }
    }

    return 'Не удалось обработать файл. Проверьте формат и попробуйте снова.';
  }

  dynamic _normalize(dynamic value) {
    if (value is Map) {
      return value.map(
        (key, val) => MapEntry(key.toString(), _normalize(val)),
      );
    }
    if (value is List) {
      return value.map(_normalize).toList();
    }
    return value;
  }

  double _asDouble(dynamic value) {
    if (value is num) {
      return value.toDouble();
    }
    return double.tryParse(value?.toString() ?? '') ?? 0.0;
  }

  String _fmtPercent(dynamic value) => '${_asDouble(value).toStringAsFixed(2)}%';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Seed Detect (офлайн)')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  ElevatedButton.icon(
                    onPressed: _state == AnalyzeState.processing
                        ? null
                        : _pickFromGallery,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Выбрать из галереи'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _state == AnalyzeState.processing
                        ? null
                        : _pickFromFiles,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('Выбрать файл'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _state == AnalyzeState.processing
                        ? null
                        : _captureFromCamera,
                    icon: const Icon(Icons.photo_camera),
                    label: const Text('Сделать снимок'),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _buildStateBlock(),
              const SizedBox(height: 16),
              if (_result != null && _result!['ok'] == true) _buildMetrics(_result!),
              if (_result != null && _result!['ok'] == true) const SizedBox(height: 16),
              if (_result != null && _result!['ok'] == true) _buildArtifacts(_result!),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStateBlock() {
    if (_state == AnalyzeState.processing || _state == AnalyzeState.picking) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Row(
            children: [
              SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 12),
              Expanded(child: Text('Выполняется локальная обработка файла...')),
            ],
          ),
        ),
      );
    }

    if (_state == AnalyzeState.error) {
      return Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            _errorMessage ?? 'Обработка завершилась ошибкой',
            style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
          ),
        ),
      );
    }

    if (_state == AnalyzeState.done) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('Готово. Метрики и артефакты показаны ниже.'),
        ),
      );
    }

    return const Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Text('Выберите один файл для локального анализа.'),
      ),
    );
  }

  Widget _buildMetrics(Map<String, dynamic> result) {
    final prep = result['preprocessing'] is Map<String, dynamic>
        ? result['preprocessing'] as Map<String, dynamic>
        : <String, dynamic>{};
    final blackArea = result['black_seed_area_px'] ?? 0;
    final allArea = result['all_seed_area_px'] ?? 0;
    final blackToAllPct = _fmtPercent(result['black_to_all_seed_ratio_pct']);
    final blackInImagePct = _fmtPercent(result['black_seed_ratio_pct']);
    final allInImagePct = _fmtPercent(result['all_seed_ratio_pct']);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Результат анализа', style: TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primaryContainer,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Ключевая метрика', style: TextStyle(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 4),
                  Text(
                    'Черные / все семечки: $blackToAllPct',
                    style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            const Text('Базовые значения площадей:'),
            Text('Площадь черных семечек: $blackArea px ($blackInImagePct от изображения)'),
            Text('Площадь всех семечек: $allArea px ($allInImagePct от изображения)'),
            const SizedBox(height: 8),
            Text('Количество всех семечек: ${result['seed_count']}'),
            Text('Количество черных семечек: ${result['black_seed_count']}'),
            Text('Размер изображения: ${result['image_w']} x ${result['image_h']}'),
            Text('Время обработки: ${result['processing_ms']} мс'),
            if (prep.isNotEmpty)
              Text(
                'Предобработка: изменение размера=${prep['resized']} масштаб=${prep['scale']}',
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildArtifacts(Map<String, dynamic> result) {
    final artifacts = result['artifacts'] is Map<String, dynamic>
        ? result['artifacts'] as Map<String, dynamic>
        : <String, dynamic>{};

    final items = <MapEntry<String, String>>[
      MapEntry('Маска всех семечек', artifacts['all_mask']?.toString() ?? ''),
      MapEntry('Оверлей всех семечек', artifacts['all_overlay']?.toString() ?? ''),
      MapEntry('Маска черных семечек', artifacts['black_mask']?.toString() ?? ''),
      MapEntry('Оверлей черных семечек', artifacts['black_overlay']?.toString() ?? ''),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: items
          .where((item) => item.value.isNotEmpty)
          .map(
            (item) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(item.key, style: const TextStyle(fontWeight: FontWeight.w700)),
                      const SizedBox(height: 8),
                      _buildImagePreview(item.key, item.value),
                      const SizedBox(height: 6),
                      Text(item.value, style: Theme.of(context).textTheme.bodySmall),
                    ],
                  ),
                ),
              ),
            ),
          )
          .toList(),
    );
  }

  void _openImageViewer(String title, String path) {
    final file = File(path);
    if (!file.existsSync()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Файл изображения не найден')),
      );
      return;
    }

    showDialog<void>(
      context: context,
      builder: (dialogContext) {
        return Dialog(
          insetPadding: const EdgeInsets.all(12),
          backgroundColor: Colors.black,
          child: SizedBox(
            width: MediaQuery.of(dialogContext).size.width,
            height: MediaQuery.of(dialogContext).size.height * 0.85,
            child: Stack(
              children: [
                Positioned.fill(
                  child: InteractiveViewer(
                    minScale: 0.5,
                    maxScale: 8,
                    child: Center(
                      child: Image.file(
                        file,
                        fit: BoxFit.contain,
                        errorBuilder: (_, __, ___) => const Text(
                          'Не удалось открыть изображение',
                          style: TextStyle(color: Colors.white),
                        ),
                      ),
                    ),
                  ),
                ),
                Positioned(
                  top: 8,
                  right: 8,
                  child: IconButton.filledTonal(
                    onPressed: () => Navigator.of(dialogContext).pop(),
                    icon: const Icon(Icons.close),
                  ),
                ),
                Positioned(
                  left: 12,
                  top: 12,
                  child: Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildImagePreview(String title, String path) {
    final file = File(path);
    if (!file.existsSync()) {
      return const Text('Файл изображения не найден');
    }

    return Material(
      borderRadius: BorderRadius.circular(8),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => _openImageViewer(title, path),
        child: Stack(
          children: [
            Image.file(
              file,
              fit: BoxFit.cover,
              width: double.infinity,
              height: 220,
              errorBuilder: (_, __, ___) => const SizedBox(
                height: 220,
                child: Center(child: Text('Не удалось отобразить изображение')),
              ),
            ),
            Positioned(
              right: 8,
              bottom: 8,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.65),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: const Text(
                  'Нажмите для увеличения',
                  style: TextStyle(color: Colors.white, fontSize: 12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
