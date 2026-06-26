# Component Counter — AI Agent Guide

## 📱 Project Overview

Android app for counting electronic components on tape/reel using camera + TensorFlow Lite object detection. Built with Jetpack Compose, CameraX, and on-device ML.

## 🏗 Architecture

```
MVVM + StateFlow + Compose
├── View (Composable) → collectAsState() from StateFlow
├── ViewModel → CounterViewModel.kt (state: DetectionState, Snapshot)
└── Model → ObjectDetectorHelper.kt (TFLite ObjectDetector wrapper)
```

### Package Structure
```
com.example.componentcounter/
├── ml/ObjectDetectorHelper.kt    — TFLite model wrapper
├── viewmodel/CounterViewModel.kt — all state management + NMS logic
├── ui/camera/CameraScreen.kt     — live camera + detection overlay
├── ui/history/HistoryScreen.kt   — snapshot list + CSV export
├── ui/main/MainScreen.kt         — legacy wrapper (keep for compat)
├── theme/                        — Color, Theme, Typography
├── MainActivity.kt               — entry point, setContent
├── Navigation.kt                 — bottom nav + tab switching
└── NavigationKeys.kt             — @Serializable nav keys
```

### Data Flow
```
Camera frame → ImageProxy → Bitmap
  → ObjectDetectorHelper.detect() → List<Detection>
  → CounterViewModel.updateDetections()
    → NMS dedup (IoU > 0.5 filter)
  → StateFlow<DetectionState> → UI collectAsState()
```

## 🔧 Build System

| Config | Value |
|--------|-------|
| compileSdk | 36 (Android 16) |
| minSdk | 24 |
| targetSdk | 36 |
| Kotlin | 2.3.20 |
| AGP | 9.0.1 |
| Java | 17 |
| Compose BOM | 2026.03.01 |

### Key Build Config
- `noCompress += "tflite"` — TFLite model files
- `buildConfig = false` — no BuildConfig generation
- Version from `version.txt` (updated by CI)
- Release signing via environment variables (`STORE_FILE`, `STORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`)

## 📦 Dependencies

### Core
- `androidx.core:core-ktx:1.18.0`
- `androidx.activity:activity-compose:1.13.0`
- Jetpack Compose BOM `2026.03.01` (UI, Material3, Tooling)

### Lifecycle & Navigation
- `lifecycle-viewmodel-compose:2.10.0`
- `lifecycle-runtime-compose:2.10.0`
- `navigation3-runtime:1.0.1` + `navigation3-ui:1.0.1`
- `lifecycle-viewmodel-navigation3:2.10.0`

### Camera (CameraX)
- `camera-core:1.4.0`
- `camera-camera2:1.4.0`
- `camera-lifecycle:1.4.0`
- `camera-view:1.4.0`

### ML
- `tensorflow-lite:2.16.1`
- `tensorflow-lite-task-vision:0.4.4` (ObjectDetector API)

### Testing
- JUnit 4.13.2
- `kotlinx-coroutines-test:1.10.2`
- Robolectric 4.14.1
- Compose UI Test JUnit4
- Roborazzi 1.14.0 (screenshot testing)

## 📐 Coding Conventions

### Kotlin
- Use `val` over `var` unless state must be mutable
- Use `object : Listener` pattern for callbacks
- Prefer `StateFlow` over LiveData
- Use `remember { mutableStateOf() }` for local UI state
- Use `collectAsState()` for ViewModel state observation

### Compose
- Screens → `@Composable fun XxxScreen(modifier: Modifier = Modifier, ...)`
- Reusable components → private `@Composable fun ...`
- No custom `@Preview` functions (use Roborazzi screenshots instead)
- Material3 components preferred

### Navigation (Nav3)
```kotlin
@Serializable data object ScreenKey : NavKey
val backStack = rememberNavBackStack(ScreenKey)
// No NavHost — manual tab switching via when(selectedTab)
```

### Imports
- Wildcard imports OK: `import androidx.compose.material3.*`
- Organize by: Android → Compose → Lifecycle → Camera → TF → project

## 🧪 Testing Strategy

### Unit Tests (`app/src/test/`)
- ViewModel tests with JUnit + coroutines test
- NMS logic tests (overlapping/non-overlapping boxes, edge cases)
- ObjectDetectorHelper error handling (Robolectric context)

### Instrumented Tests (`app/src/androidTest/`)
- Compose UI tests for screen rendering
- Roborazzi screenshot tests for visual regression
- Permission flow tests

### CI Checks
- `./gradlew test` — unit tests **must pass**
- `./gradlew lint` — no errors
- `./gradlew detekt` — no issues
- `./gradlew verifyRoborazziDebug` — screenshot comparison

## 🤖 CI/CD Workflows

### review.yml (PR → detekt + lint + test)
Trigger: `pull_request: [opened, synchronize, reopened]`

### qa.yml (PR + main → screenshot + test)
Trigger: `pull_request` + `push: main`

### release.yml (push main → build + sign + release)
Trigger: `push: main` (paths-ignore: *.md, .gitignore)
Jobs: version → build (test + assemble) → release (GitHub Release + APK)

## 📝 Commit Convention

Conventional Commits — CI reads commit msg for version bump:

| Prefix | Version Bump |
|--------|-------------|
| `BREAKING CHANGE:` / `feat!:`, `fix!:`, `refactor!:` | **Major** |
| `feat:` | **Minor** |
| `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `ci:` | **Patch** |

## ⚠️ Common Gotchas

1. **Nav3 API** — Not standard Navigation Compose. Uses `rememberNavBackStack()` + manual tab switching via `when(selectedTab)`. No NavHost. NavKeys must be `@Serializable data object`.
2. **TFLite Model** — `mobilenetv1.tflite` loaded from assets. Uses `ObjectDetector.createFromFileAndOptions()`. Model has 2 threads, 0.4 threshold, 100 max results.
3. **CameraX Lifecycle** — Camera bound to lifecycle in `AndroidView factory`. When `lensFacing` changes, rebind happens automatically because factory re-runs. Camera executor is single-thread.
4. **Snapshot persistence** — Currently in-memory only. Data lost on process death. History screen shows `LazyColumn` of cards with timestamp + count + inference time.
5. **Image format** — Camera analyzer uses `OUTPUT_IMAGE_FORMAT_RGBA_8888` with fallback to YUV_420_888 → NV21 → JPEG → Bitmap conversion.
6. **Release APK** — Requires signing via env vars. CI decodes keystore from `KEYSTORE_BASE64` secret.
