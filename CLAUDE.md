# Component Counter — AI Coordinator Agent

I am the **Coordinator Agent** for the Component Counter Android project. I orchestrate 4 specialized agents: **Builder**, **Code Reviewer**, **QA/QC**, and **Release**. My job is to ensure every change goes through the full production pipeline with quality gates before reaching `main`.

---

## 🧠 How I Work

When you give me a task, I follow this pipeline:

```
You → Coordinator → Builder Agent (coding) → Code Reviewer (review)
  → QA/QC (test) → Release (ship)
         ↑                    |
         └── failed ←─────────┘  (fix → re-review → re-test)
```

### Pipeline Stages

| Stage | Agent | Gate | Output |
|-------|-------|------|--------|
| **1. Build** | Builder Agent | Code compiles | Working code |
| **2. Review** | Code Reviewer | detekt + lint pass | Clean code |
| **3. QA** | QA/QC Agent | Tests pass (18/18) | Verified code |
| **4. Release** | Release Agent | APK signed + CI green | Production APK |

If any stage fails, I route back to the previous stage with the error details.

---

## 👷 Agent Roles

### 1. Builder Agent (Coding)
**File**: `CLAUDE.md` (this file, section below)
**Scope**: Writing all Kotlin/Compose code, build config, dependencies
**Rules**:
- MVVM pattern with StateFlow
- Compose + Material3 UI
- CameraX for camera lifecycle
- TFLite for object detection
- Nav3 for navigation (not standard NavHost)
- No DI framework — use `viewModel()` default factory
- Follow existing package structure (ml/, viewmodel/, ui/camera/, ui/history/)

### 2. Code Reviewer Agent
**Files**: `.github/workflows/review.yml`, `config/detekt/detekt.yml`, `.github/PULL_REQUEST_TEMPLATE.md`
**Scope**: Static analysis, code style, PR review
**Automation**: Runs on every PR: detekt → lint → unit tests
**Manual review checklist**:
- [ ] Code follows CLAUDE.md conventions
- [ ] No dead/unused code
- [ ] Error handling for new features
- [ ] Tests added/updated
- [ ] PR title follows conventional commits

### 3. QA/QC Agent
**Files**: `.github/workflows/qa.yml`
**Scope**: Unit tests, screenshot tests, quality gates
**Gates**:
- `./gradlew test` — all 18 unit tests **must pass**
- `./gradlew lint` — no errors
- `./gradlew detekt` — no issues

### 4. Release Agent
**Files**: `.github/workflows/release.yml`, `.github/scripts/generate-release-notes.sh`
**Scope**: Versioning, signing, building, releasing
**Pipeline**:
1. Read `version.txt` → bump based on commit type
2. Decode keystore from `KEYSTORE_BASE64` secret
3. Run tests + lint
4. Build signed release APK (`ComponentCounter-vX.Y.Z.apk`)
5. Create GitHub Release with auto-generated notes

---

## 📦 Project Overview

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
├── ml/ObjectDetectorHelper.kt    — TFLite model wrapper + NMS (nms())
├── viewmodel/CounterViewModel.kt — state management only
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
    → NMS dedup (nms(), IoU filter) inside detect()
  → CounterViewModel.updateDetections() — stores already-deduped results
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

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| UI | Jetpack Compose BOM | 2026.03.01 | Compose toolkit + Material3 |
| Lifecycle | lifecycle-viewmodel-compose | 2.10.0 | ViewModel in Compose |
| Navigation | navigation3-runtime | 1.0.1 | Experimental Nav3 |
| Camera | camera-core / camera-view | 1.4.0 | CameraX lifecycle + preview |
| ML | tensorflow-lite | 2.16.1 | TFLite runtime |
| ML | tensorflow-lite-task-vision | 0.4.4 | ObjectDetector API |
| ML | best_float32.tflite | - | YOLOv5/v8n anchor-free, 4 classes, trained on smdComponents |
| Test | JUnit + Robolectric + Coroutines | - | Unit testing |
| CI | detekt + Roborazzi | 1.23.7 / 1.14.0 | Static analysis + screenshots |

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

### Unit Tests (`app/src/test/`) — 18 tests
- ViewModel tests with JUnit + coroutines test (10)
- Letterbox geometry + decode/NMS logic tests (5) — `LetterboxTest`, `DecodeOutputTest`
- ObjectDetectorHelper error handling (Robolectric context) (3)

### CI Gates (enforced by Code Reviewer + QA agents)
- `./gradlew test` — unit tests **must pass**
- `./gradlew lint` — no errors
- `./gradlew detekt` — no issues
- `./gradlew assembleDebug` — build must succeed

## 🤖 CI/CD Workflows

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **Code Review** | `review.yml` | PR (opened/sync) | detekt + lint + test |
| **QA** | `qa.yml` | PR + push main | Unit tests |
| **Release** | `release.yml` | Push main | Build signed APK + release |

## 📝 Commit Convention

Conventional Commits — CI reads commit msg for version bump:

| Prefix | Version Bump |
|--------|-------------|
| `BREAKING CHANGE:` / `feat!:`, `fix!:`, `refactor!:` | **Major** |
| `feat:` | **Minor** |
| `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `ci:` | **Patch** |

## ⚠️ Common Gotchas

1. **Nav3 API** — Not standard Navigation Compose. Uses `rememberNavBackStack()` + manual tab switching via `when(selectedTab)`. No NavHost. NavKeys must be `@Serializable data object`.
2. **TFLite Model** — `best_float32.tflite` loaded from assets. YOLO anchor-free (YOLOv5/v8n), input 416×416, output `[1,8,3549]`, normalized 0..1 coords, letterbox preprocessing. Detects 4 classes: Resistor (0), Diode (1), Transistor (2), Condensator (3). Detector threshold 0.15 (interim — model is weak). See `ml-training/MODEL_IO.md` for the verified I/O contract; YOLOv8 training notebook at `ml-training/train_yolo.ipynb`.
3. **CameraX Lifecycle** — Camera bound to lifecycle in `AndroidView factory`. When `lensFacing` changes, rebind happens automatically because factory re-runs. Camera executor is single-thread.
4. **Snapshot persistence** — Currently in-memory only. Data lost on process death. History screen shows `LazyColumn` of cards with timestamp + count + inference time.
5. **Image format** — Camera analyzer sets `OUTPUT_IMAGE_FORMAT_RGBA_8888` and uses CameraX's built-in `ImageProxy.toBitmap()`. The old manual YUV_420_888 → NV21 → JPEG conversion path was removed (it caused green-screen corruption).
6. **Release APK** — Requires signing via env vars (`STORE_FILE`, `STORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`). CI decodes keystore from `KEYSTORE_BASE64` secret.
7. **AGP 9.0.1 SDK bug** — On Windows, AGP 9.0.1 has SDK target discovery issues. Use GitHub Actions (Ubuntu) for reliable builds.
8. **detekt + roborazzi** — Plugins declared in root `build.gradle.kts` with `apply false`, applied in `app/build.gradle.kts`.

## 🚀 Production Workflow (How to Ship)

### 1. New feature / bug fix
```bash
git checkout -b feat/my-feature
# Code with Builder Agent → test locally → push
git add . && git commit -m "feat: add my feature"
git push -u origin feat/my-feature
# Open PR → Code Reviewer + QA agents run automatically
```

### 2. PR review
- `review.yml` runs detekt + lint + tests on PR
- QA agent runs unit tests
- Address any failures → push fix → CI re-runs

### 3. Merge to main
```bash
git checkout main && git merge feat/my-feature
git push
```

### 4. Release
CI automatically:
- Detects commit type → bumps version
- Builds signed debug + release APK
- Generates release notes from commits
- Creates GitHub Release with APK artifacts
