import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:seed_detect_app/main.dart';

const _channel = MethodChannel('seed_detect/analyzer');

void main() {
  testWidgets('App renders title', (tester) async {
    await tester.pumpWidget(const SeedDetectApp());
    expect(find.text('Seed Detect (офлайн)'), findsOneWidget);
    expect(find.text('Сделать снимок'), findsOneWidget);
  });

  testWidgets('Shows localized message for pdf_multiple_images', (tester) async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(_channel, (call) async {
      if (call.method == 'analyzeImage') {
        return {
          'ok': false,
          'error': {
            'code': 'pdf_multiple_images',
            'message': 'PDF contains multiple images',
          },
        };
      }
      if (call.method == 'saveImageToGallery') {
        return {
          'ok': true,
          'savedUri': 'content://mock/saved',
        };
      }
      return null;
    });

    await tester.pumpWidget(const SeedDetectApp());
    final state = tester.state(find.byType(AnalyzeScreen));
    await (state as dynamic).runAnalysisForTest('file:///mock/input.pdf');
    await tester.pumpAndSettle();

    expect(
      find.text('В PDF найдено несколько изображений. Нужен PDF ровно с одним изображением.'),
      findsOneWidget,
    );

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(_channel, null);
  });
}
