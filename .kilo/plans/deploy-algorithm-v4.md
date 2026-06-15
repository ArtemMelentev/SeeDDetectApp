# План внедрения алгоритма v4 (VersionWIthBetterDetection)

## Источник

`AlgorithmVersions/VersionWIthBetterDetection/run_pipeline_v2.py` (1958 строк)

## Что изменилось относительно текущей v3

Сравнительный анализ `run_pipeline_v2.py` ↔ `segment_seeds_scan.py` выявил 4 содержательных различия:

### 1. `_refine_quad_by_border_color` — linefit-рефайнмент
Standalone содержит продвинутую версию с edge-line fitting через перспективную трансформацию:
- Внутренние хелперы: `_first_edge_run`, `_collect_horizontal_edges`, `_collect_vertical_edges`, `_robust_fit_line`, `_fit_horizontal_edge`, `_fit_vertical_edge`
- Сначала пытается найти точные линии краёв бумаги (`_linefit_rect_quad`)
- При неудаче — fallback на базовый rectangle-trim (текущая v3-логика)
- Текущая интеграционная версия: только упрощённый rectangle-trim без linefit

### 2. `_segment_black_seeds_v2` — warm-edge фиксация
Standalone добавляет вычисление `a_chan`, `b_chan`, `chroma` и использует их в фильтрации bright_edge:
- `warm_edge = (b_chan > 15.0) & (chroma > 15.0) & (light > l_core - 10.0)`
- `bright_edge = ((light > l_core + 12.0) | warm_edge) & (grown > 0) & (interior == 0)`
- Текущая версия: только `bright_edge = (light > l_core + 12.0) & (grown > 0) & (interior == 0)`

### 3. `_recover_black_pixels` — дополнительный фильтр
Standalone добавляет раннюю отбраковку: `if fill < 0.18 or aspect < 0.06: continue` перед существующими проверками `fill < 0.30` и `aspect < 0.18`.

### 4. `_fixup_masks` — возвращает info-dict
Standalone возвращает `tuple[np.ndarray, np.ndarray, dict]` с метаинформацией (`paper_L`, `all_added_px`, `black_added_px`). Текущая версия — только `tuple[np.ndarray, np.ndarray]`.

## Функции без изменений (НЕ трогать)

Следующие функции **идентичны** между standalone и integration — портировать не нужно:
`_odd`, `_fill_holes`, `_largest_component`, `_component_bbox`, `_score_paper_component`, `_order_quad`, `_quad_from_contour`, `_line_from_points`, `_intersect_lines`, `_quad_from_paper_edges`, `_moving_average_1d`, `_first_paper_run`, `_cyan_field_mask`, `_table_mask`, `_detect_paper_mask`, `_warp_to_paper`, `_resize_short`, `_estimate_paper_color`, `_paper_pixel_mask`, `_fit_poly_flatfield`, `_estimate_background`, `_paper_wb_and_levels`, `_normalize_scan`, `_compute_delta_e`, `_white_balance_to_paper`, `_segment_all_seeds_v2`, `_interior_holes`

## Шаги внедрения

### Шаг 1: Архивировать исходник
- Создать `AlgorithmVersions/VersionV4/`
- Скопировать `run_pipeline_v2.py` → `AlgorithmVersions/VersionV4/run_pipeline_v4.py`
- Валидация: `python -c "import ast; ast.parse(open('AlgorithmVersions/VersionV4/run_pipeline_v4.py').read())"`

### Шаг 2: Переименовать старые заменяемые функции в `_legacy`
В `segment_seeds_scan.py`:
- `_refine_quad_by_border_color` → `_refine_quad_by_border_color_legacy`
- `_recover_black_pixels` → `_recover_black_pixels_legacy`
- `_fixup_masks` → `_fixup_masks_legacy`
- `_segment_black_seeds_v2` — **оставить как есть** (v2 — отдельная версия, не legacy)

### Шаг 3: Добавить новые v4-хелперы в секцию `# v3 helpers`
Обновить заголовок на `# v4 helpers (see AlgorithmVersions/VersionV4/run_pipeline_v4.py)`

Добавить после существующих хелперов:
1. `_refine_quad_by_border_color_v4` — с linefit edge detection + legacy fallback
2. `_segment_black_seeds_v4` — с warm-edge chroma-based фильтрацией
3. `_recover_black_pixels_v4` — с дополнительным fill/aspect фильтром
4. `_fixup_masks_v4` — возвращает `(all, black, info_dict)`

### Шаг 4: Обновить pipeline в `analyze_image()`
Заменить вызовы:
- `_segment_black_seeds_v2` → `_segment_black_seeds_v4`
- `_recover_black_pixels` внутри `_fixup_masks_v4` → `_recover_black_pixels_v4`
- `_fixup_masks` → `_fixup_masks_v4` (распаковать 3-tuple: `all_mask, blk_mask, fixup_info`)

`_refine_quad_by_border_color_v4` вызывается внутри `_detect_paper_mask` — обновить вызов там.

### Шаг 5: Обновить дефолты и комментарии
- Строка ≈24: `# v4 pipeline defaults (see AlgorithmVersions/VersionV4/run_pipeline_v4.py)`
- Строка ≈225: `# v4 helpers (see AlgorithmVersions/VersionV4/run_pipeline_v4.py)`
- `DEFAULT_*` — без изменений (параметры те же)

### Шаг 6: Проверить тестами
```bash
python -m unittest -q
```

## Неприкасаемое
- `android_bridge.py`
- Публичная сигнатура `analyze_image()`
- Ключи payload
- Имена артефактов
- Коды ошибок
- `release_mode` логика
- `_safe_cv2_read` / `_safe_cv2_write`
- Тесты `test_segment_contract.py`
