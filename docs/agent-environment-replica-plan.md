# План для агента: перенос рабочего окружения 1:1 на другой ПК

## Цель
Поднять на новом компьютере максимально идентичное окружение для `C:\SeeDDetectApp`, чтобы продолжить работу без смены контекста: тот же git-срез, те же пути, те же инструменты и тот же запуск приложения.

## Эталон (снято с текущего ПК)
- ОС: `Windows 10.0.26200.8246`.
- Репозиторий: `C:\SeeDDetectApp`.
- Git remote: `https://github.com/ArtemMelentev/SeeDDetectApp.git`.
- Ветка/состояние: `main...origin/main`, рабочее дерево чистое.
- HEAD: `25c2220877361d4b20e8d22cb819ee5b360e11aa`.
- Flutter: `3.22.3` (Dart `3.4.4`), путь `C:\Users\Artem Melentyev\source\flutter`.
- JDK: Temurin `17.0.18+8`, путь `C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot`.
- Python: `3.10.11` и `3.11.9`.
- Android SDK: `C:\Android\Sdk`; AVD: `SeedDetect_API34`.
- Kilo global config: `C:\Users\Artem Melentyev\.config\kilo\kilo.jsonc`.
- Важный внешний путь из Kilo-конфига: `C:\Scanfex\ScanfexGraceLessons\.kilocode\rules\rules.md`.

## Инструкции агенту

### 1) На исходном ПК: собрать пакет миграции
1. Создай папку `C:\migration\seeddetect-YYYYMMDD-HHMM`.
2. Сохрани снимок состояния:
   - `git -C C:\SeeDDetectApp status --porcelain=v1 --branch > git-status.txt`
   - `git -C C:\SeeDDetectApp rev-parse HEAD > git-head.txt`
   - `git -C C:\SeeDDetectApp stash list > git-stash.txt`
3. Сохрани версии инструментов:
   - `git --version > versions.txt`
   - `java -version >> versions.txt 2>&1`
   - `py -3.10 --version >> versions.txt`
   - `py -3.11 --version >> versions.txt`
   - `"C:\Users\Artem Melentyev\source\flutter\bin\flutter.bat" --version >> versions.txt`
   - `"C:\Android\Sdk\platform-tools\adb.exe" --version >> versions.txt`
   - `"C:\Android\Sdk\emulator\emulator.exe" -version >> versions.txt`
4. Экспортируй список пакетов Windows:
   - `winget export -o winget-export.json --include-versions`.
5. Скопируй в пакет миграции:
   - `C:\Users\Artem Melentyev\.config\kilo\` (полностью).
   - `C:\Scanfex\ScanfexGraceLessons\.kilocode\rules\rules.md`.
   - `C:\SeeDDetectApp\docs\setup-summary.md`.
   - `C:\SeeDDetectApp\.kilo\plans\1776249736022-proud-comet.md`.
6. Если на момент миграции дерево не чистое, добавь:
   - `git -C C:\SeeDDetectApp diff > working-tree.patch`
   - `git -C C:\SeeDDetectApp diff --staged > staged.patch`
   - архив неотслеживаемых файлов.

### 2) На новом ПК: восстановить базовую систему и пути
1. Создай одинаковые директории:
   - `C:\SeeDDetectApp`
   - `C:\Android\Sdk`
   - `C:\Users\Artem Melentyev\source\flutter`
   - `C:\Scanfex\ScanfexGraceLessons\.kilocode\rules`
2. Установи инструменты (строго с проверкой версий):
   - Python 3.10 и 3.11.
   - Temurin JDK 17.
   - Android Studio и/или Android cmdline-tools.
   - Git.
3. Установи Flutter в тот же путь и зафиксируй версию `3.22.3` (допустимо фиксировать по commit hash из `flutter --version`).
4. Для Android SDK установи пакеты:
   - `platform-tools`
   - `emulator`
   - `platforms;android-34`
   - `platforms;android-36`
   - `build-tools;34.0.0`
   - `build-tools;36.0.0`
   - `build-tools;28.0.3`
   - `system-images;android-34;google_apis;x86_64`
5. Создай AVD:
   - `SeedDetect_API34` с образом `system-images;android-34;google_apis;x86_64`.

### 3) Восстановить Kilo и агентные правила
1. Скопируй `kilo.jsonc`, `package.json`, `bun.lock` и нужные каталоги из бэкапа в `C:\Users\Artem Melentyev\.config\kilo\`.
2. Восстанови `C:\Scanfex\ScanfexGraceLessons\.kilocode\rules\rules.md` в точно тот же путь.
3. Проверь `C:\Users\Artem Melentyev\.config\kilo\kilo.jsonc`:
   - модель `openai/gpt-5.3-codex`;
   - ссылка на `instructions` указывает на существующий `rules.md`.
4. Если имя пользователя на новом ПК другое, обнови абсолютные пути в:
   - `C:\SeeDDetectApp\.vscode\settings.json`
   - `C:\SeeDDetectApp\.vscode\launch.json`
   - `C:\Users\<NEW_USER>\.config\kilo\kilo.jsonc`

### 4) Восстановить проект в той же точке
1. Клонируй репозиторий в `C:\SeeDDetectApp`.
2. Переключись на `main`, подтяни состояние и проверь commit:
   - `git checkout main`
   - `git pull --ff-only`
   - `git rev-parse HEAD` должен вернуть `25c2220877361d4b20e8d22cb819ee5b360e11aa`
   - если commit отличается, выполни `git checkout 25c2220877361d4b20e8d22cb819ee5b360e11aa` и создай рабочую ветку от этого commit.
3. Если есть patch-файлы из шага 1, примени их в том же порядке.
4. Восстанови локальные артефакты контекста:
   - `.kilo/plans/1776249736022-proud-comet.md`.

### 5) Восстановить зависимости проекта
1. Flutter:
   - `C:\Users\Artem Melentyev\source\flutter\bin\flutter.bat pub get`
2. Python для локальных запусков:
   - `py -3.10 -m pip install -r C:\SeeDDetectApp\requirements.txt`
3. Kilo project-level плагины:
   - `npm --prefix C:\SeeDDetectApp\.kilo install`

### 6) Проверка идентичности окружения
1. Git:
   - `git -C C:\SeeDDetectApp rev-parse HEAD` == `25c2220877361d4b20e8d22cb819ee5b360e11aa`
   - `git -C C:\SeeDDetectApp status --porcelain=v1 --branch` показывает чистое дерево.
2. Flutter/Android:
   - `"C:\Users\Artem Melentyev\source\flutter\bin\flutter.bat" --version` -> `3.22.3` / Dart `3.4.4`.
   - `"C:\Android\Sdk\emulator\emulator.exe" -list-avds` содержит `SeedDetect_API34`.
3. VS Code конфиги:
   - `dart.flutterSdkPath` указывает на корректный путь.
   - `ANDROID_HOME`, `ANDROID_SDK_ROOT`, `JAVA_HOME` выставляются как в `C:\SeeDDetectApp\.vscode\settings.json`.
4. Kilo:
   - глобальный конфиг читается без ошибок;
   - файл `rules.md` доступен по пути из `kilo.jsonc`.

### 7) Старт с текущего места
1. Открой проект `C:\SeeDDetectApp`.
2. Открой план `C:\SeeDDetectApp\.kilo\plans\1776249736022-proud-comet.md`.
3. Запусти `SeedDetect (One Click)` из `C:\SeeDDetectApp\.vscode\launch.json`.

## Критерий готовности
- Новый ПК повторяет исходный стек по ключевым версиям и путям.
- Репозиторий открыт на том же commit и в чистом состоянии.
- AVD `SeedDetect_API34` доступен.
- Агент может продолжить работу с контекстом из текущего плана без дополнительной ручной подготовки.
