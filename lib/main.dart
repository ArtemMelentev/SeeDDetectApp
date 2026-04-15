import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/services.dart';

const _channel = MethodChannel('seed_detect/analyzer');

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

class _AnalyzeScreenState extends State<AnalyzeScreen> {
  final ImagePicker _imagePicker = ImagePicker();

  AnalyzeState _state = AnalyzeState.idle;
  Map<String, dynamic>? _result;
  String? _errorMessage;

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
      allowedExtensions: const ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
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
        final error = normalized['error'];
        final message = error is Map<String, dynamic>
            ? (error['message']?.toString() ?? 'Unknown error')
            : 'Unknown error';
        setState(() {
          _state = AnalyzeState.error;
          _result = normalized;
          _errorMessage = message;
        });
      }
    } on PlatformException catch (exc) {
      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = exc.message ?? 'Platform error';
      });
    } catch (exc) {
      setState(() {
        _state = AnalyzeState.error;
        _errorMessage = exc.toString();
      });
    }
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Seed Detect Offline')),
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
                    label: const Text('Select from gallery'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _state == AnalyzeState.processing
                        ? null
                        : _pickFromFiles,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('Select from files'),
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
              Expanded(child: Text('Processing image locally on device...')),
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
            _errorMessage ?? 'Processing failed',
            style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
          ),
        ),
      );
    }

    if (_state == AnalyzeState.done) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('Done. Metrics and overlays are shown below.'),
        ),
      );
    }

    return const Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Text('Pick one file and run local analysis.'),
      ),
    );
  }

  Widget _buildMetrics(Map<String, dynamic> result) {
    final prep = result['preprocessing'] is Map<String, dynamic>
        ? result['preprocessing'] as Map<String, dynamic>
        : <String, dynamic>{};

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Metrics', style: TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Text('All seeds count: ${result['seed_count']}'),
            Text('All seeds area (px): ${result['all_seed_area_px']}'),
            Text('Black seeds count: ${result['black_seed_count']}'),
            Text('Black seeds area (px): ${result['black_seed_area_px']}'),
            Text('Image size: ${result['image_w']} x ${result['image_h']}'),
            Text('Processing time: ${result['processing_ms']} ms'),
            if (prep.isNotEmpty)
              Text(
                'Preprocess: resized=${prep['resized']} scale=${prep['scale']}',
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
      MapEntry('All mask', artifacts['all_mask']?.toString() ?? ''),
      MapEntry('All overlay', artifacts['all_overlay']?.toString() ?? ''),
      MapEntry('Black mask', artifacts['black_mask']?.toString() ?? ''),
      MapEntry('Black overlay', artifacts['black_overlay']?.toString() ?? ''),
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
                      _buildImagePreview(item.value),
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

  Widget _buildImagePreview(String path) {
    final file = File(path);
    if (!file.existsSync()) {
      return const Text('Image file not found');
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.file(
        file,
        fit: BoxFit.cover,
        width: double.infinity,
        height: 220,
        errorBuilder: (_, __, ___) => const Text('Unable to display image'),
      ),
    );
  }
}
