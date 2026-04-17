import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:seed_detect_app/main.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Analyze screen shows picker buttons', (tester) async {
    await tester.pumpWidget(const SeedDetectApp());
    await tester.pumpAndSettle();

    expect(find.text('Выбрать из галереи'), findsOneWidget);
    expect(find.text('Выбрать файл'), findsOneWidget);
  });
}
