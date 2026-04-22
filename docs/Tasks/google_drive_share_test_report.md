# Отчет по тест-сценариям Google Drive Share

## Контекст
- Реализация: `ShareReceiverActivity` + `SharedContentImporter` + интеграция с `MainActivity`/Flutter.
- Тип проверки в текущей среде: статическая проверка кода + `flutter test`.
- Ручные сценарии на устройстве: требуют отдельного прогона QA на Android-девайсе.

## Статус сценариев

| Сценарий | Статус | Комментарий |
|---|---|---|
| Google Drive -> Отправить копию -> PDF -> приложение видно -> файл открыт | READY_FOR_QA | Манифест и обработка `ACTION_SEND` для `application/pdf` добавлены, требуется ручная проверка на устройстве |
| Google Drive -> Отправить копию -> JPEG -> приложение видно -> файл открыт | READY_FOR_QA | Манифест и обработка `ACTION_SEND` для `image/*` добавлены, требуется ручная проверка на устройстве |
| `SEND_MULTIPLE` для изображений | READY_FOR_QA | Поддержка `ACTION_SEND_MULTIPLE` реализована, открывается первый импортированный файл |
| Большой файл | READY_FOR_QA | Импорт через поток `ContentResolver.openInputStream`, требуется проверка производительности на устройстве |
| Файл без расширения, но с корректным MIME | READY_FOR_QA | MIME берется из `ContentResolver.getType` с fallback |
| Неподдерживаемый MIME | READY_FOR_QA | Валидация MIME и контролируемая ошибка реализованы |
| Поврежденный/недоступный URI | READY_FOR_QA | Добавлены обработчики `SecurityException`, `FileNotFoundException`, `IOException` |

## Автоматическая проверка
- `flutter test`: выполняется в рамках локальной верификации после изменений.
