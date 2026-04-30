# Окружение проекта (требования и установка)

## Где записаны требования

1. Flutter/Dart зависимости (пакеты):
- `pubspec.yaml` (что нужно)
- `pubspec.lock` (точные версии после `flutter pub get`)

2. Android/Gradle зависимости и плагины:
- `android/settings.gradle` (версии плагинов, `flutter.sdk` из `android/local.properties`)
- `android/build.gradle` (AGP/Chaquopy classpath)
- `android/gradle/wrapper/gradle-wrapper.properties` (версия Gradle Wrapper)

3. Требования к SDK/инструментам (toolchain):
- `toolchain.yaml` (версии и минимум для старта)

4. Python зависимости (локальные утилиты/скрипты, не Chaquopy):
- `requirements.txt`

## Файлы-шаблоны для локальной машины

- `android/local.properties.example`
  - скопировать в `android/local.properties`
  - заполнить `flutter.sdk` и `sdk.dir`
  - файл `android/local.properties` обычно не коммитится

- `android/key.properties.example`
  - скопировать в `android/key.properties`
  - заполнить параметры signing
  - файл `android/key.properties` не коммитится (секреты)

## Быстрый старт (Windows)

1. Установить инструменты:
- Flutter SDK версии из `toolchain.yaml`
- Android SDK (platform-tools + emulator + нужные packages из `toolchain.yaml`)
- JDK 17
- Python 3.10 (нужен для Chaquopy; в Gradle используется `py -3.10`)

2. Настроить локальные пути:
- либо добавить `flutter`, `adb`, `emulator` в `PATH`
- либо заполнить `android/local.properties` по шаблону

3. Установить Dart/Flutter зависимости:
```powershell
flutter pub get
```

4. Установить локальные Python-зависимости (если нужны скрипты вне Android/Chaquopy):
```powershell
python -m pip install -r requirements.txt
```

## Установка инструментов (Windows, примеры команд)

Примечание: команды ниже приведены как ориентир. Актуальные версии см. в `toolchain.yaml`.

```powershell
# JDK 17
winget install --id EclipseAdoptium.Temurin.17.JDK -e --silent --accept-package-agreements --accept-source-agreements

# Python 3.10 (нужен для Chaquopy)
winget install --id Python.Python.3.10 -e --silent --accept-package-agreements --accept-source-agreements

# Android Studio (вместе с SDK Manager)
winget install --id Google.AndroidStudio -e --silent --accept-package-agreements --accept-source-agreements
```

Android SDK packages (если используешь `sdkmanager`):
```powershell
# Пример: после установки cmdline-tools, укажи реальный путь к SDK
$env:ANDROID_SDK_ROOT = 'C:\\path\\to\\Android\\Sdk'

& "$env:ANDROID_SDK_ROOT\\cmdline-tools\\latest\\bin\\sdkmanager.bat" --sdk_root=$env:ANDROID_SDK_ROOT --licenses
& "$env:ANDROID_SDK_ROOT\\cmdline-tools\\latest\\bin\\sdkmanager.bat" --sdk_root=$env:ANDROID_SDK_ROOT `
  "platform-tools" "emulator" "platforms;android-34" "build-tools;34.0.0" "system-images;android-34;google_apis;x86_64"

# AVD
& "$env:ANDROID_SDK_ROOT\\cmdline-tools\\latest\\bin\\avdmanager.bat" create avd --name SeedDetect_API34 --package "system-images;android-34;google_apis;x86_64" --device "pixel" --force
```

## VS Code задачи (Android)

- `Start SeedDetect emulator`
  - требует доступных `adb` и `emulator` (через `ANDROID_SDK_ROOT`/`ANDROID_HOME`, или `android/local.properties` (`sdk.dir`), или `PATH`)
  - пытается использовать AVD `SeedDetect_API34`, иначе берёт первый доступный

- `Run SeedDetect debug (emulator)`
  - требует `flutter` (через `android/local.properties` (`flutter.sdk`) или `PATH`)

- `Build SeedDetect release APK (phone)`
  - требует `android/key.properties` (signing)
  - требует `flutter`
