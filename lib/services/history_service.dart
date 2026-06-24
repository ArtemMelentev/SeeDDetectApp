import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:path_provider/path_provider.dart';

class HistoryEntry {
  final String id;
  final String timestamp;
  final String originalFileName;
  final String thumbnailPath;
  final int seedCount;
  final int blackSeedCount;
  final double blackToAllRatio;

  HistoryEntry({
    required this.id,
    required this.timestamp,
    required this.originalFileName,
    required this.thumbnailPath,
    required this.seedCount,
    required this.blackSeedCount,
    required this.blackToAllRatio,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'timestamp': timestamp,
        'originalFileName': originalFileName,
        'thumbnailPath': thumbnailPath,
        'seedCount': seedCount,
        'blackSeedCount': blackSeedCount,
        'blackToAllRatio': blackToAllRatio,
      };

  factory HistoryEntry.fromJson(Map<String, dynamic> json) => HistoryEntry(
        id: json['id'] as String,
        timestamp: json['timestamp'] as String,
        originalFileName: json['originalFileName'] as String,
        thumbnailPath: json['thumbnailPath'] as String,
        seedCount: json['seedCount'] as int,
        blackSeedCount: json['blackSeedCount'] as int,
        blackToAllRatio: (json['blackToAllRatio'] as num).toDouble(),
      );
}

class HistoryService {
  Directory? _historyDir;

  Future<Directory> getHistoryDir() async {
    if (_historyDir != null) return _historyDir!;
    final appDoc = await getApplicationDocumentsDirectory();
    _historyDir = Directory('${appDoc.path}${Platform.pathSeparator}history');
    if (!_historyDir!.existsSync()) {
      _historyDir!.createSync(recursive: true);
    }
    return _historyDir!;
  }

  Future<String> get appDocPath async {
    final dir = await getApplicationDocumentsDirectory();
    return dir.path;
  }

  Future<File> _indexFile() async {
    final dir = await getHistoryDir();
    return File('${dir.path}${Platform.pathSeparator}index.json');
  }

  String _generateScanId() {
    final now = DateTime.now();
    final datePart =
        '${now.year}${_pad(now.month)}${_pad(now.day)}';
    final random = Random().nextInt(0xFFFFFFF);
    final uid = random.toRadixString(36).padLeft(7, '0');
    return '${datePart}_$uid';
  }

  String _pad(int n) => n.toString().padLeft(2, '0');

  String _isoNow() => DateTime.now().toIso8601String();

  Future<List<HistoryEntry>> loadIndex() async {
    final file = await _indexFile();
    if (!file.existsSync()) return [];

    final basePath = (await getApplicationDocumentsDirectory()).path;

    try {
      final content = await file.readAsString();
      final list = json.decode(content) as List<dynamic>;
      final entries = <HistoryEntry>[];
      for (final e in list) {
        try {
          final entry =
              HistoryEntry.fromJson(e as Map<String, dynamic>);
          final scanDir = Directory(
            '$basePath${Platform.pathSeparator}history${Platform.pathSeparator}${entry.id}',
          );
          if (scanDir.existsSync()) {
            entries.add(entry);
          }
        } catch (_) {
          // skip malformed entry
        }
      }
      entries.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      return entries;
    } catch (_) {
      return [];
    }
  }

  Future<void> _writeIndex(List<HistoryEntry> entries) async {
    final file = await _indexFile();
    final list = entries.map((e) => e.toJson()).toList();
    await file.writeAsString(json.encode(list));
  }

  Future<Map<String, dynamic>> loadMetadata(String scanId) async {
    final dir = await getHistoryDir();
    final file = File(
      '${dir.path}${Platform.pathSeparator}$scanId${Platform.pathSeparator}metadata.json',
    );
    if (!file.existsSync()) return {};
    try {
      final content = await file.readAsString();
      return json.decode(content) as Map<String, dynamic>;
    } catch (_) {
      return {};
    }
  }

  Future<bool> saveScan(Map<String, dynamic> result) async {
    String? scanId;
    try {
      final historyDir = await getHistoryDir();
      scanId = _generateScanId();
      final scanDir = Directory(
        '${historyDir.path}${Platform.pathSeparator}$scanId',
      );
      scanDir.createSync(recursive: true);

      final originalPath = result['image']?.toString();
      if (originalPath != null && originalPath.isNotEmpty) {
        final src = File(originalPath);
        if (src.existsSync()) {
          await src.copy(
            '${scanDir.path}${Platform.pathSeparator}original.jpg',
          );
        }
      }

      final artifacts = result['artifacts'] is Map<String, dynamic>
          ? result['artifacts'] as Map<String, dynamic>
          : <String, dynamic>{};
      final overlayPath = artifacts['black_overlay']?.toString();

      var thumbnailSaved = false;
      if (overlayPath != null && overlayPath.isNotEmpty) {
        final src = File(overlayPath);
        if (src.existsSync()) {
          await src.copy(
            '${scanDir.path}${Platform.pathSeparator}overlay.jpg',
          );
          try {
            await _generateThumbnail(
              '${scanDir.path}${Platform.pathSeparator}overlay.jpg',
              '${scanDir.path}${Platform.pathSeparator}thumbnail.jpg',
            );
            thumbnailSaved = true;
          } catch (_) {
            // fallback to overlay
          }
        }
      }

      final originalFileName = result['originalFileName']?.toString() ??
          result['image']
              ?.toString()
              .split(Platform.pathSeparator)
              .last ??
          'unknown';

      final metrics = {
        'seed_count': result['seed_count'],
        'black_seed_count': result['black_seed_count'],
        'all_seed_area_px': result['all_seed_area_px'],
        'image_area_px': result['image_area_px'],
        'black_seed_area_px': result['black_seed_area_px'],
        'all_seed_ratio_pct': result['all_seed_ratio_pct'],
        'black_seed_ratio_pct': result['black_seed_ratio_pct'],
        'black_to_all_seed_ratio_pct':
            result['black_to_all_seed_ratio_pct'],
        'processing_ms': result['processing_ms'],
        'image_w': result['image_w'],
        'image_h': result['image_h'],
        'preprocessing': result['preprocessing'] ?? {},
      };

      final metadata = {
        'id': scanId,
        'timestamp': _isoNow(),
        'originalFileName': originalFileName,
        'metrics': metrics,
        'files': {
          'original': 'original.jpg',
          'overlay': 'overlay.jpg',
          'thumbnail': thumbnailSaved ? 'thumbnail.jpg' : null,
        },
      };

      await File(
        '${scanDir.path}${Platform.pathSeparator}metadata.json',
      ).writeAsString(json.encode(metadata));

      final basePath = (await getApplicationDocumentsDirectory()).path;
      final entry = HistoryEntry(
        id: scanId,
        timestamp: _isoNow(),
        originalFileName: originalFileName,
        thumbnailPath:
            '$basePath${Platform.pathSeparator}history${Platform.pathSeparator}$scanId${Platform.pathSeparator}${thumbnailSaved ? 'thumbnail.jpg' : 'overlay.jpg'}',
        seedCount: _asInt(result['seed_count']),
        blackSeedCount: _asInt(result['black_seed_count']),
        blackToAllRatio: _asDouble(
          result['black_to_all_seed_ratio_pct'],
        ),
      );

      final entries = await loadIndex();
      entries.insert(0, entry);
      await _writeIndex(entries);

      return true;
    } catch (e) {
      if (scanId != null) {
        try {
          final historyDir = await getHistoryDir();
          final scanDir = Directory(
            '${historyDir.path}${Platform.pathSeparator}$scanId',
          );
          if (scanDir.existsSync()) {
            scanDir.deleteSync(recursive: true);
          }
        } catch (_) {}
      }
      return false;
    }
  }

  Future<bool> deleteEntry(String scanId) async {
    try {
      final historyDir = await getHistoryDir();
      final scanDir = Directory(
        '${historyDir.path}${Platform.pathSeparator}$scanId',
      );
      if (scanDir.existsSync()) {
        scanDir.deleteSync(recursive: true);
      }
      final entries =
          (await loadIndex()).where((e) => e.id != scanId).toList();
      await _writeIndex(entries);
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> deleteAll() async {
    try {
      final historyDir = await getHistoryDir();
      if (historyDir.existsSync()) {
        historyDir.deleteSync(recursive: true);
        historyDir.createSync(recursive: true);
      }
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<void> _generateThumbnail(
    String srcPath,
    String destPath,
  ) async {
    final bytes = await File(srcPath).readAsBytes();
    final codec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: 200,
    );
    final frame = await codec.getNextFrame();
    final image = frame.image;

    final aspectRatio = image.width / image.height;
    const targetWidth = 200;
    final targetHeight = (200 / aspectRatio).round();

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(
        0,
        0,
        image.width.toDouble(),
        image.height.toDouble(),
      ),
      ui.Rect.fromLTWH(
        0,
        0,
        targetWidth.toDouble(),
        targetHeight.toDouble(),
      ),
      ui.Paint(),
    );
    final picture = recorder.endRecording();
    final resized =
        await picture.toImage(targetWidth, targetHeight);
    final byteData =
        await resized.toByteData(format: ui.ImageByteFormat.png);
    if (byteData != null) {
      await File(destPath)
          .writeAsBytes(byteData.buffer.asUint8List());
    }
  }

  int _asInt(dynamic v) =>
      v is int ? v : int.tryParse(v?.toString() ?? '') ?? 0;

  double _asDouble(dynamic v) =>
      v is double ? v : double.tryParse(v?.toString() ?? '') ?? 0.0;
}
