# Windows Build Roadmap

Status: **Phase 1 implemented** (this PR) | Last updated: 2026-04-03

## Goal

Full Windows build parity with Linux/macOS across CI, release, and Python packaging — **without modifying any C++ or Python source code**. The codebase already compiles cleanly on MSVC thanks to existing `#ifdef _WIN32` / `#ifdef _MSC_VER` guards.

---

## Phase 1: CI Green on Windows (this PR)

All changes are in `.github/workflows/ci.yml` only.

| Task | Status |
|------|--------|
| Install Eigen3 via chocolatey in `cxx_core_build` Windows step | Done |
| Add `windows-latest` to `pure_python_results` matrix | Done |
| Add `windows-latest` to `python_bindings_smoke` matrix | Done |
| Use cross-platform `PYTHONPATH` env var (not bash inline) | Done |
| Use YAML `>` folded scalar for long pytest invocations (avoids `\` escaping) | Done |

### What already worked before this PR

- `cxx_core_build` / `windows-msvc` matrix entry (configure + build + ctest)
- `release.yml` Windows artifact packaging (`.zip` with `.exe` binaries)
- All MSVC compile definitions: `NOMINMAX`, `_USE_MATH_DEFINES`, `_CRT_SECURE_NO_WARNINGS`
- SIMD dispatch: `/arch:AVX2` and `/arch:AVX512` on MSVC
- `/MT` static CRT linkage on FlexAID, FlexAIDdS, tENCoM
- `gtest_force_shared_crt ON` for GoogleTest compatibility
- Eigen3 FetchContent fallback (slow but functional)

---

## Phase 2: OpenMP on Windows (future)

Currently `FLEXAIDS_USE_OPENMP=OFF` on Windows CI because MSVC's bundled OpenMP is v2.0 (2002-era, no `#pragma omp simd`). Options:

| Approach | Effort | Tradeoff |
|----------|--------|----------|
| **Use LLVM OpenMP via vcpkg** | Medium | `vcpkg install llvm-openmp` + toolchain file; full OpenMP 4.5+ |
| **Use Intel oneAPI compiler** | Medium | `icx` supports OpenMP 5.x natively; add CI matrix entry |
| **Keep OpenMP OFF on MSVC** | None | Single-threaded on Windows; acceptable for testing |

Recommendation: keep OpenMP OFF for CI testing (validates correctness), enable in release builds via vcpkg if users request it.

---

## Phase 3: vcpkg Integration (future)

Replace ad-hoc `choco install` with a `vcpkg.json` manifest for reproducible Windows dependency management.

```json
{
  "name": "flexaidds",
  "version-string": "1.76",
  "dependencies": [
    "eigen3",
    { "name": "openmp", "platform": "windows" },
    { "name": "pybind11", "platform": "windows" }
  ]
}
```

Benefits:
- Deterministic dependency versions across CI runs
- Easier local Windows development (`cmake --preset windows-release`)
- CURL for DatasetRunner HTTP downloads (currently shell-out to `curl.exe`)

---

## Phase 4: CMake Presets (future)

Add `CMakePresets.json` with named configurations for all platforms:

```json
{
  "configurePresets": [
    {
      "name": "windows-release",
      "generator": "Ninja",
      "binaryDir": "build",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "FLEXAIDS_USE_OPENMP": "OFF",
        "FLEXAIDS_USE_CUDA": "OFF",
        "FLEXAIDS_USE_METAL": "OFF",
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      },
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows" }
    }
  ]
}
```

CI steps simplify to: `cmake --preset windows-release && cmake --build build`

---

## Phase 5: Windows Python Wheel Distribution (future)

Publish pre-built `_core.pyd` wheels for Windows so users don't need a C++ compiler.

| Task | Description |
|------|-------------|
| Add `cibuildwheel` to CI | Build `cp39-cp312` wheels on `windows-latest` |
| Configure `pyproject.toml` | Add `[tool.cibuildwheel]` with MSVC settings |
| Publish to PyPI | `twine upload` on tag push |

---

## Phase 6: Windows ARM64 (future, low priority)

Windows on ARM (Snapdragon X) is emerging. Current status:
- MSVC supports ARM64 cross-compilation
- No AVX2/AVX512 on ARM64 (NEON only) — SIMD dispatch already handles this
- Eigen3 has ARM64 NEON vectorization
- No CI runner available yet (`windows-arm64` not in GitHub Actions)

Action: monitor GitHub Actions runner availability; add matrix entry when available.

---

## Existing MSVC Guards in Source Code

These are already in place and require no changes:

| File | Guard | Purpose |
|------|-------|---------|
| `gaboom.cpp:46` | `#ifdef _WIN32` | `<windows.h>` vs `<unistd.h>` |
| `gaboom.cpp:1769` | `#ifdef _WIN32` | `Sleep()` vs `usleep()` |
| `DatasetRunner.cpp:224` | `#ifdef _MSC_VER` | `_popen`/`_pclose` vs `popen`/`pclose` |
| `top.cpp`, `read_input.cpp` | `#ifdef _WIN32` | Path separator `\\` vs `/` |
| `hardware_detect.cpp` | `#ifdef _MSC_VER` | `<intrin.h>` + `__cpuid` vs `<cpuid.h>` |
| `simd_distance.h` | `#ifdef _MSC_VER` | `__restrict` vs `__restrict__` |
| `flexaid.h` | `#ifdef _MSC_VER` | `_CRT_SECURE_NO_WARNINGS` |
