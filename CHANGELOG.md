# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-19

### Added

- Initial release of torch-amd-setup
- Complete README with quick start and usage examples
- `scripts/verify_gpu.py` - GPU verification and diagnostics
- `scripts/benchmark.py` - Performance benchmarking vs CPU
- `scripts/hello_gpu.py` - Simple demo script for end-to-end testing
- `scripts/setup_env.py` - Interactive environment setup wizard with `--dry-run` and `--silent` flags
- `requirements.txt` - Dependency management (torch-directml, diffusers, transformers, accelerate)
- `run.bat` - Windows menu launcher for scripts
- `.gitignore` - Python, IDE, and output artifacts
- `LICENSE` - MIT License
- `INSTALL.md` - Detailed installation guide with troubleshooting
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - This file
- `.github/workflows/ci.yml` - CI/CD pipeline (Python 3.10+, Windows/Ubuntu/macOS)
- `.github/workflows/changelog.yml` - Automated changelog updates
- `.github/ISSUE_TEMPLATE/bug_report.md` - Bug report template
- `.github/ISSUE_TEMPLATE/feature_request.md` - Feature request template

### Features

- Full torch-directml support for AMD/Intel/Nvidia GPUs on Windows
- Tensor operations and neural network training on GPU
- Matrix multiplication and element-wise operation benchmarks
- VRAM estimation and monitoring
- Environment validation (Python version, driver checks)
- Comprehensive error messages with fix suggestions

### Documentation

- Quick start guide in README
- Usage examples for tensor ops, device placement, and model training
- Limitations and known issues documented
- Common pitfalls and fixes in INSTALL.md
- GPU driver installation instructions
- Performance tips and VRAM management

## Future Plans

- [ ] ROCm support for Linux
- [ ] Multi-GPU training utilities
- [ ] Model zoo with pre-optimized examples
- [ ] Performance profiling tools
- [ ] Docker containerization
- [ ] Integration with popular frameworks

---

For more information, visit [torch-amd-setup GitHub](https://github.com/ChharithOeun/torch-amd-setup)

Support the project:
[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20Me%20A%20Coffee-FFDD00?style=flat&logoColor=white)](https://buymeacoffee.com/chharith)
