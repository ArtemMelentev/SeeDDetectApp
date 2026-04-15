# Summary: установка окружения для SeedDetectApp (Windows)

- Flutter SDK: `C:\Users\Artem Melentyev\source\flutter` (установлен через `git clone`, использована версия `3.22.3`).
- Android Studio: `C:\Program Files\Android\Android Studio` (установлена через `winget`).
- Android SDK root: `C:\Android\Sdk` (перенесен из `%LOCALAPPDATA%` в путь без пробелов).
- Android SDK cmdline-tools: `C:\Android\Sdk\cmdline-tools\latest`.
- Android Platform-Tools (`adb`): `C:\Android\Sdk\platform-tools`.
- Android Emulator: `C:\Android\Sdk\emulator`.
- Android system image (эмулятор): `system-images;android-34;google_apis;x86_64`.
- AVD: `SeedDetect_API34` (эмулятор API 34, x86_64).
- JDK 17 (Temurin): `C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot`.
- Python 3.11: `C:\Users\Artem Melentyev\AppData\Local\Programs\Python\Python311` (через `winget`).
- Python 3.10: `C:\Users\Artem Melentyev\AppData\Local\Programs\Python\Python310` (через `winget`, нужен для Chaquopy в текущей конфигурации).

## Установленные Android SDK пакеты

- `platform-tools`
- `emulator`
- `platforms;android-34`
- `platforms;android-36`
- `build-tools;34.0.0`
- `build-tools;36.0.0`
- `build-tools;28.0.3`
- `system-images;android-34;google_apis;x86_64`
- `ndk;28.2.13676358` (дотянут автоматически Gradle во время сборки)

## Переменные окружения (использовались при запуске)

- `ANDROID_HOME=C:\Android\Sdk`
- `ANDROID_SDK_ROOT=C:\Android\Sdk`
- `JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot`

Примечание: в текущей системе `flutter` не добавлен в глобальный `PATH`, запуск выполнялся через полный путь `C:\Users\Artem Melentyev\source\flutter\bin\flutter.bat`.

## Что изменено в проекте для сборки

- Добавлен Gradle Wrapper:
  - `android/gradlew`
  - `android/gradlew.bat`
  - `android/gradle/wrapper/gradle-wrapper.properties`
  - `android/gradle/wrapper/gradle-wrapper.jar`
- Обновлена конфигурация Android/Chaquopy:
  - `android/build.gradle`
  - `android/app/build.gradle`
- Обновлены заметки:
  - `README.md`
  - `.gitignore`

## Текущее состояние запуска

- `flutter doctor -v`: Android toolchain корректный, лицензии приняты.
- Эмулятор `SeedDetect_API34` запускается, `adb` видит `emulator-5554`.
- Блокер сборки `flutter run`: сетевой таймаут при скачивании wheel из `https://chaquo.com/pypi-13.1` (Chaquopy dependency download timeout).

## Команды, которые можно повторить на другом ПК

```powershell
# 1) Python
winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
winget install --id Python.Python.3.10 -e --silent --accept-package-agreements --accept-source-agreements

# 2) JDK 17
winget install --id EclipseAdoptium.Temurin.17.JDK -e --silent --accept-package-agreements --accept-source-agreements

# 3) Android Studio и platform-tools
winget install --id Google.AndroidStudio -e --silent --accept-package-agreements --accept-source-agreements
winget install --id Google.PlatformTools -e --silent --accept-package-agreements --accept-source-agreements

# 4) Flutter SDK
git clone https://github.com/flutter/flutter.git -b stable C:\Users\<USER>\source\flutter
cd C:\Users\<USER>\source\flutter
git checkout 3.22.3

# 5) Android cmdline-tools (пример в C:\Android\Sdk)
# скачать: https://dl.google.com/android/repository/commandlinetools-win-13114758_latest.zip
# распаковать в: C:\Android\Sdk\cmdline-tools\latest

# 6) Лицензии + SDK пакеты
C:\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat --sdk_root=C:\Android\Sdk --licenses
C:\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat --sdk_root=C:\Android\Sdk "platform-tools" "platforms;android-34" "platforms;android-36" "build-tools;34.0.0" "build-tools;36.0.0" "build-tools;28.0.3" "emulator" "system-images;android-34;google_apis;x86_64"

# 7) Создание AVD
C:\Android\Sdk\cmdline-tools\latest\bin\avdmanager.bat create avd --name SeedDetect_API34 --package "system-images;android-34;google_apis;x86_64" --device "pixel" --force

# 8) Проверка проекта
cd C:\SeeDDetectApp
C:\Users\<USER>\source\flutter\bin\flutter.bat pub get
C:\Users\<USER>\source\flutter\bin\flutter.bat doctor -v
```
