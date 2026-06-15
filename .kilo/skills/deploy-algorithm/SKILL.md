---
name: deploy-algorithm
description: Внедрить новую версию алгоритма сегментации семян из standalone-файла в приложение
---

# Deploy Algorithm Skill

Автоматизирует портирование новой версии алгоритма сегментации из standalone Python-файла (из `AlgorithmVersions/`) в интеграционный модуль `segment_seeds_scan.py` приложения SeeDDetect.

## Входные данные

Переменная `$ARGUMENTS` содержит всё, что пользователь передал после `/deploy-algorithm`:

```
/deploy-algorithm <путь_к_файлу_алгоритма> [название_новой_версии]
```

- `<путь_к_файлу_алгоритма>` — обязательный, путь к standalone `.py` файлу с новой версией алгоритма.
- `[название_новой_версии]` — опционально, например `v4`. Если не указан — извлечь из имени файла или уточнить.

**Если `$ARGUMENTS` пуст** — запросить у пользователя путь к файлу алгоритма и название версии.

## Ключевые контракты и правила

### Неприкасаемое (никогда не менять)

| Контракт | Детали |
|----------|--------|
| `android_bridge.py` | `android/app/src/main/python/android_bridge.py` — не трогать |
| Публичная сигнатура `analyze_image()` | `analyze_image(input_path, output_dir, max_pixels=..., max_side=..., do_shading=..., target_short=..., release_mode=...) -> dict` |
| Ключи payload | `ok`, `image`, `image_w`, `image_h`, `processing_ms`, `seed_count`, `all_seed_area_px`, `image_area_px`, `black_seed_count`, `black_seed_area_px`, `all_seed_ratio_pct`, `black_seed_ratio_pct`, `black_to_all_seed_ratio_pct`, `preprocessing`, `artifacts` |
| Имена артефактов | `<stem>_all_mask.png`, `<stem>_all_overlay.jpg`, `<stem>_black_mask.png`, `<stem>_black_overlay.jpg` |
| Коды ошибок | `file_not_found`, `invalid_input`, `unsupported_format`, `decode_error`, `opencv_error`, `write_error`, `unexpected_error` |
| Логика `release_mode` | При `release_mode=True`: только `black_overlay` с крестиками. При `False`: все 4 артефакта |
| Unicode-safe I/O | `_safe_cv2_read` / `_safe_cv2_write` через `np.fromfile` / `tofile` |
| Тесты | `tests/test_segment_contract.py` — должны проходить без изменений (4 тест-кейса) |

### Что портировать

Только **вычислительные хелперы** из standalone-файла:
- Функции без GUI, без CLI, без batch-runner, без tkinter
- Вспомогательные функции: геометрия, маски, нормализация, сегментация, визуализация
- Pipeline-логика: последовательность вызовов в `analyze_image()`

### Что НЕ портировать

- GUI-код (tkinter, matplotlib-окна, интерактивные элементы)
- CLI-код (argparse, `if __name__ == "__main__"`, batch-обработчики)
- Логгеры/прогресс-бары/UI-колбэки
- Всё, что относится к запуску standalone, а не к вычислениям

### Именование функций

- **Старые функции, которые заменяются** → переименовать с суффиксом `_legacy` (например, `_segment_all_seeds_legacy`)
- **Новые функции** → префикс `_` + суффикс `_v{N}` (например, `_detect_paper_mask_v4`, `_segment_all_seeds_v4`)
- **Функции, логика которых НЕ изменилась** — оставить текущее имя (например, `_segment_all_seeds_v2` остаётся `_v2`, даже если используется в v4)
- **Новый суффикс только для реально новых или изменённых функций**

### Fallback

- Для каждой новой критичной операции (детект бумаги, warp, кроп) **обязателен fallback** на предыдущий метод
- При ошибке/деградации fallback должен возвращать работоспособный результат без падения пайплайна
- Пример: v3 warp-to-paper с fallback на v2 bbox-crop
- Пример: детект бумаги с fallback на full-frame mask

### Git-коммиты

- Коммиты на **русском языке**
- Паттерн: «внедрил новую версию»

## Процесс внедрения (6 шагов)

### Шаг 1: Архивировать исходник в `AlgorithmVersions/`

1. Скопировать standalone-файл в `AlgorithmVersions/Version{N}/` (создать директорию).
2. Название директории — новая версия (например, `VersionV4`). Если не указано — уточнить.
3. Проверить читаемость: `python -c "import ast; ast.parse(open('<путь>').read())"`.

### Шаг 2: Проанализировать новый алгоритм (карта функций)

Прочитать standalone-файл и составить карту:

1. **Общие хелперы** (odd, fill_holes, largest_component, component_bbox, resize_short, ...)
2. **Детект бумаги** (detect_paper_mask и все подфункции: score, refine, quad, ...)
3. **ROI-извлечение** (crop, warp, или новый метод)
4. **Нормализация** (normalize_scan, white_balance, shading, ...)
5. **Сегментация** (segment_all_seeds, segment_black_seeds, compute_delta_e, ...)
6. **Fixup** (fixup_masks, recover_black_pixels, interior_holes, ...)
7. **Визуализация** (visualize_all, visualize_black, overlay, ...)
8. **Pipeline** (как функции вызываются в главном блоке/раннере)

Пометить для каждой функции:
- **ПОРТ** — вычислительная, нужна в интеграции, логика изменилась/новая
- **ЕСТЬ** — аналог уже существует в `segment_seeds_scan.py` с той же логикой
- **ПРОПУСК** — GUI/CLI/batch, не портировать

### Шаг 3: Портировать вычислительные функции в `segment_seeds_scan.py`

#### Структурные ориентиры в `segment_seeds_scan.py`

| Строка (≈) | Секция | Что делать |
|------------|--------|------------|
| ≈24 | `# v3 pipeline defaults` | Обновить дефолты и комментарий |
| ≈29 | `# Legacy scan segmentation` | Переименовывать заменяемые функции в `_legacy` |
| ≈225 | `# v3 helpers` | **Вставлять новые хелперы сюда**, обновить заголовок секции (`# v{N} helpers`) |
| ≈1453 | `# Visualization helpers` | Добавлять/обновлять визуализацию |
| ≈1500 | `_safe_cv2_read` / I/O | Не трогать |
| ≈1569 | `def analyze_image(` | Сигнатура — не менять публичный контракт |
| ≈1604 | `try:` блок pipeline | **Заменить pipeline на новую версию** |

**Важно:** номера строк даны с `≈` — они сдвигаются при редактировании. Ориентироваться на комментарии секций, а не на точные номера.

#### Правила вставки

1. Новые функции вставлять в секцию хелперов (≈строка 225), группируя по назначению: общие → детект бумаги → ROI → нормализация → сегментация → fixup.
2. Если функция уже существует с той же логикой — не дублировать.
3. Публичные имена из standalone (без `_`) переименовать в приватные (с `_`): `odd` → `_odd`, `fill_holes` → `_fill_holes`.
4. Удалить из портированных функций все обращения к GUI, логгерам, progress-барам, tkinter.
5. Визуализацию добавлять в секцию Visualization helpers (≈строка 1453), соблюдая сигнатуру `_visualize_*_v{N}()`.

### Шаг 4: Обновить pipeline в `analyze_image()`

1. Найти блок `try:` (≈строка 1604) внутри `analyze_image()`.
2. Заменить последовательность вызовов на новый pipeline.
3. Сохранить fallback-логику для критичных операций.
4. Не менять:
   - Формирование метрик (проценты, `_safe_percent`)
   - Запись артефактов (пути, имена файлов)
   - Ветку `release_mode`
   - Блоки `except cv2.error` и `except Exception`
5. Убедиться, что все используемые функции определены выше.

### Шаг 5: Обновить дефолты и комментарии

1. Обновить строку ≈24: `# v{N} pipeline defaults (see AlgorithmVersions/Version{VN}/run_pipeline_v{N}.py)`.
2. Обновить строку ≈225: `# v{N} helpers (...)`.
3. Если версия меняет дефолтные параметры — обновить `DEFAULT_*`.
4. Заменённые старые функции → переименовать в `_legacy`, переместить в Legacy-секцию.
5. Обновить docstring в `analyze_image()` если pipeline описан.

### Шаг 6: Проверить тестами

```bash
python -m unittest -q
```

Все 4 тест-кейса должны пройти:
- `test_returns_structured_payload_and_artifacts`
- `test_returns_not_found_error`
- `test_photo_with_paper_background_produces_masks`
- `test_release_mode_writes_only_black_overlay`

Если тесты падают:
1. Проверить, что публичный контракт `analyze_image()` не нарушен.
2. Проверить fallback-логику — возможно, новый метод падает на синтетических тестовых изображениях.
3. Ослабить fallback: если площадь мала или quad деградирует → full-frame/bbox-crop.
4. **Не менять тесты** без крайней необходимости.

## Чеклист самопроверки

- [ ] Исходный standalone-файл скопирован в `AlgorithmVersions/Version{N}/`
- [ ] Все вычислительные хелперы портированы с правильным неймингом
- [ ] GUI/CLI/batch-раннер НЕ портированы
- [ ] Старые заменённые функции переименованы в `_legacy`
- [ ] Fallback есть для каждой новой критичной операции
- [ ] Комментарии секций и дефолтов обновлены
- [ ] `android_bridge.py` НЕ тронут
- [ ] Публичная сигнатура `analyze_image()` не изменена
- [ ] Коды ошибок не изменены
- [ ] Имена артефактов не изменены
- [ ] `release_mode` логика сохранена (ветки `if release_mode:` и `else`)
- [ ] Unicode-safe I/O (`_safe_cv2_read`/`_safe_cv2_write`) сохранён
- [ ] `python -m unittest -q` проходит (4/4)
- [ ] Нет импортов tkinter, argparse, threading, queue из портированного кода (если не используются)

## Пример внедрения: v2 → v3

### Исходный файл
`AlgorithmVersions/VersionWithoutScan2/run_pipeline_v3.py` (1667 строк)

### Что изменилось относительно v2
- Детект бумаги возвращает не только маску, но и 4 угла (`quad`)
- Вместо bbox-кропа — perspective warp (`warp_to_paper`)
- Fallback: если warp падает → bbox-crop из v2

### Какие функции портированы
- `_odd`, `_fill_holes`, `_largest_component`, `_component_bbox` — уже были (переиспользованы)
- `_score_paper_component`, `_quad_from_paper_edges`, `_moving_average_1d`, `_first_paper_run`, `_refine_quad_by_border_color`, `_cyan_field_mask`, `_table_mask` — новые подфункции детекта
- `_detect_paper_mask` — заменён на v3-версию (возвращает `info["quad"]`)
- `_order_quad`, `_quad_from_contour`, `_line_from_points`, `_intersect_lines` — геометрия quad
- `_warp_to_paper` — новый метод ROI
- `_crop_to_paper` — оставлен как fallback
- `_segment_all_seeds_v2`, `_segment_black_seeds_v2` — логика НЕ изменилась, имена `_v2`
- `_visualize_all_v2`, `_visualize_black_v2`, `_visualize_black_release` — визуализация

### Pipeline в `analyze_image()` (v3)
```python
paper_full, detect_info = _detect_paper_mask(prepared_img)
quad = np.asarray(detect_info.get("quad"), dtype=np.float32).reshape(4, 2)
try:
    crop_img, crop_paper, warp_info = _warp_to_paper(prepared_img, paper_full, quad)
    if out_w < 80 or out_h < 80:
        raise ValueError(...)
except Exception:
    crop_img, crop_paper, _crop = _crop_to_paper(prepared_img, paper_full)
# upscale → normalize → wb → segment → fixup → metrics → artifacts
```
