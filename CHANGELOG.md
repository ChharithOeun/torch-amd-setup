# Changelog — torch-amd-setup

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) |
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

Auto-updated by `scripts/update_changelog.py` — run on every commit via git hook,
and by GitHub Actions on every push to `main`.

> **Verification policy:** No guessing. No fabricated numbers.
> Benchmarks marked ✅ were measured on real hardware with documented specs.
> Claims marked 🔬 require a hardware run to confirm.

---

## [Unreleased] — updated 2026-03-29

### Changed
- Add neon banner (`0e99f9c`)


## [0.3.0] — 2026-03-23

### Added

- **Diffusers DirectML device string trap** — documented Session 3 discovery:
  `pipe.to("privateuseone:0")` silently falls back to CPU in diffusers.
  Correct approach: pass the device *object* (`pipe.to(dml.device())`), never the string.
  This is the #1 silent performance killer on Windows DirectML setups.
- `docs/lessons-learned.md` Session 3 — Diffusers-specific DirectML pitfalls
- `docs/troubleshooting.md` — 14 real-production failure entries (all from hardware runs)

### Fixed

- N/A — documentation-only release

---

## [0.2.0] — 2026-03-23

### Added

- **Real DirectML benchmarks on RX 5700 XT** ✅ (Windows 11, Python 3.11.9,
  torch 2.4.1, torch-directml 0.2.5, AMD Ryzen 7 host)
  - CPU baseline: 250.4 ms, 0.55 TFLOPS
  - AMD DirectML float32: 6.2 ms, 22.04 TFLOPS → **40.2× speedup** ✅
  - Workload: float32 matrix multiply (512×512, batch=32, 100 warmup + 100 timed)
- `docs/lessons-learned.md` Session 2 — 5 Windows-specific discoveries:
  1. `torch-directml` compiled against Python 3.11 ABI — hard ceiling, 3.12+ silently fails
  2. Install DirectML *before* torch — pip resolver doesn't back-solve torch 2.4.1
  3. Device string is `privateuseone:0` not `dml:0` — expected, normal
  4. CTranslate2 (faster-whisper) has zero DirectML support — Whisper stays on CPU ⚠️
  5. Ollama ignores `OLLAMA_MODELS` env var if set after parent process starts

### Changed

- `README.md` — benchmark table added, Known Limitations section added

---

## [0.1.0] — 2026-03-23

### Added

- Initial release: **AMD GPU auto-detection for PyTorch**
- `torch_amd_setup/detect.py` — priority detection chain:
  CUDA → ROCm → DirectML → MPS → CPU
- `GFX_OVERRIDE_MAP` — `gfx1010` (RX 5700 XT) maps to `HSA_OVERRIDE_GFX_VERSION=10.3.0`,
  required because gfx1010 is not in ROCm's default hardware allow-list
- `get_best_device()` → returns device type string
- `get_torch_device()` → returns `torch.device` object (correct for diffusers `.to()` calls)
- `get_dtype()` → float16 for CUDA/ROCm, float32 for DirectML/MPS/CPU
- `device_info()` → full diagnostics dict (device, dtype, driver, python version, etc.)
- CLI: `python -m torch_amd_setup.detect` — standalone diagnostics
- `examples/basic_usage.py` — end-to-end usage with forward pass verification
- `docs/lessons-learned.md` Session 1 — 5 ROCm Linux lessons:
  1. `HSA_OVERRIDE_GFX_VERSION=10.3.0` required for gfx1010 (RX 5700 XT)
  2. Ubuntu ships `rocminfo` v5.0.0 stubs — must evict before installing ROCm 6.1
  3. `set -e` + `grep` with no match = silent script death (exit code 1)
  4. fairseq2n CPU binary missing from PyPI — Seamless M4T install fails without it
  5. numpy 2.x ABI break — use `numpy<2.0` with torch ≤2.4
- `docs/troubleshooting.md` — initial 9 entries from Session 1
- `pyproject.toml` — pip-installable package

---

_Auto-updated by `scripts/update_changelog.py`. Run `python scripts/update_changelog.py` after committing._
