import 'package:flutter_test/flutter_test.dart';

import 'package:seed_detect_app/main.dart';

void main() {
  testWidgets('App renders title', (tester) async {
    await tester.pumpWidget(const SeedDetectApp());
    expect(find.text('Seed Detect Offline'), findsOneWidget);
  });
}
