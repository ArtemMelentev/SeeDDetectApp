import 'dart:io';

import 'package:flutter/material.dart';

import '../services/history_service.dart';

class HistoryScreen extends StatefulWidget {
  final HistoryService historyService;

  const HistoryScreen({super.key, required this.historyService});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<HistoryEntry> _entries = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() => _loading = true);
    final entries = await widget.historyService.loadIndex();
    if (!mounted) return;
    setState(() {
      _entries = entries;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('История'),
        actions: [
          if (_entries.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_sweep),
              onPressed: _confirmClearAll,
            ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_entries.isEmpty) {
      return _buildEmptyState();
    }
    return _buildHistoryList();
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.history,
            size: 80,
            color: Theme.of(context).disabledColor,
          ),
          const SizedBox(height: 16),
          Text(
            'История сканов пуста',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 8),
          const Text('После анализа результаты появятся здесь'),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () => Navigator.of(context).pop(),
            icon: const Icon(Icons.search),
            label: const Text('К анализу'),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryList() {
    return ListView.builder(
      padding: const EdgeInsets.all(8),
      itemCount: _entries.length,
      itemBuilder: (context, index) {
        final entry = _entries[index];
        return Dismissible(
          key: ValueKey(entry.id),
          direction: DismissDirection.endToStart,
          confirmDismiss: (_) => _confirmDelete(entry),
          background: Container(
            alignment: Alignment.centerRight,
            padding: const EdgeInsets.only(right: 20),
            color: Colors.red,
            child: const Icon(Icons.delete, color: Colors.white),
          ),
          child: _buildScanCard(entry),
        );
      },
    );
  }

  Widget _buildScanCard(HistoryEntry entry) {
    final thumbFile = File(entry.thumbnailPath);
    final hasThumb = thumbFile.existsSync();

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 4),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () => _openDetail(entry),
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: SizedBox(
                  width: 80,
                  height: 80,
                  child: hasThumb
                      ? Image.file(
                          thumbFile,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) =>
                              const Icon(Icons.image_not_supported),
                        )
                      : const Icon(Icons.image_not_supported),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      entry.originalFileName,
                      style: const TextStyle(
                        fontWeight: FontWeight.w600,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _fmtDate(entry.timestamp),
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Всего: ${entry.seedCount}  ·  Чёрных: ${entry.blackSeedCount} (${entry.blackToAllRatio.toStringAsFixed(1)}%)',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }

  String _fmtDate(String iso) {
    try {
      final dt = DateTime.parse(iso);
      final day = dt.day.toString().padLeft(2, '0');
      final month = dt.month.toString().padLeft(2, '0');
      final year = dt.year;
      final hour = dt.hour.toString().padLeft(2, '0');
      final minute = dt.minute.toString().padLeft(2, '0');
      return '$day.$month.$year $hour:$minute';
    } catch (_) {
      return iso;
    }
  }

  Future<bool> _confirmDelete(HistoryEntry entry) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Удалить запись?'),
        content: Text(
          'Запись от ${_fmtDate(entry.timestamp)} будет удалена из истории.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Отмена'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Удалить'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await widget.historyService.deleteEntry(entry.id);
      await _loadHistory();
      if (!mounted) return false;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Запись удалена')),
      );
    }
    return false;
  }

  Future<void> _confirmClearAll() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Очистить всю историю?'),
        content: const Text('Все записи будут безвозвратно удалены.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Отмена'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Очистить'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await widget.historyService.deleteAll();
      await _loadHistory();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('История очищена')),
      );
    }
  }

  Future<void> _openDetail(HistoryEntry entry) async {
    final basePath = (await widget.historyService.appDocPath);
    final metadata = await widget.historyService.loadMetadata(entry.id);

    if (!mounted) return;

    if (!context.mounted) return;

    final overlayFile = File(
      '$basePath${Platform.pathSeparator}history${Platform.pathSeparator}${entry.id}${Platform.pathSeparator}overlay.jpg',
    );

    showDialog<void>(
      context: context,
      builder: (dialogContext) {
        final metrics = metadata['metrics'] as Map<String, dynamic>? ?? {};

        return Dialog(
          insetPadding: const EdgeInsets.all(8),
          backgroundColor: Colors.black,
          child: Column(
            children: [
              Expanded(
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: InteractiveViewer(
                        minScale: 0.5,
                        maxScale: 8,
                        child: Center(
                          child: overlayFile.existsSync()
                              ? Image.file(
                                  overlayFile,
                                  fit: BoxFit.contain,
                                  errorBuilder: (_, __, ___) =>
                                      const Text(
                                    'Не удалось открыть изображение',
                                    style: TextStyle(color: Colors.white),
                                  ),
                                )
                              : const Text(
                                  'Файл не найден',
                                  style: TextStyle(color: Colors.white),
                                ),
                        ),
                      ),
                    ),
                    Positioned(
                      top: 8,
                      right: 8,
                      child: IconButton.filledTonal(
                        onPressed: () =>
                            Navigator.of(dialogContext).pop(),
                        icon: const Icon(Icons.close),
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                color: Colors.grey[900],
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      entry.originalFileName,
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _fmtDate(entry.timestamp),
                      style: TextStyle(
                        color: Colors.grey[400],
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 8),
                    _metricRow(
                      'Всего семечек',
                      '${metrics['seed_count'] ?? entry.seedCount}',
                    ),
                    _metricRow(
                      'Чёрных семечек',
                      '${metrics['black_seed_count'] ?? entry.blackSeedCount}',
                    ),
                    _metricRow(
                      'Чёрные / все',
                      _fmtPct(metrics['black_to_all_seed_ratio_pct'] ?? entry.blackToAllRatio),
                    ),
                    _metricRow(
                      'Площадь всех семечек',
                      '${metrics['all_seed_area_px'] ?? '-'} px',
                    ),
                    _metricRow(
                      'Площадь чёрных',
                      '${metrics['black_seed_area_px'] ?? '-'} px',
                    ),
                    _metricRow(
                      'Время обработки',
                      '${metrics['processing_ms'] ?? '-'} мс',
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _metricRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(color: Colors.grey[400], fontSize: 13),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }

  String _fmtPct(dynamic v) {
    if (v is num) return '${v.toStringAsFixed(2)}%';
    return '${v ?? '-'}%';
  }
}
