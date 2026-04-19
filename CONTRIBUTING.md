# Contributing to torch-amd-setup

Thank you for your interest in contributing! This project aims to make PyTorch on AMD GPU Windows as accessible and robust as possible.

## Ways to Contribute

### 1. Report Bugs

Found a bug? Open a [GitHub Issue](https://github.com/ChharithOeun/torch-amd-setup/issues/new?template=bug_report.md) with:

- GPU model (AMD, Intel, Nvidia)
- GPU driver version
- Python version
- torch-directml version (from `python -c "import torch_directml; print(torch_directml.__version__)"`)
- PyTorch version (from `python -c "import torch; print(torch.__version__)"`)
- Full error message and traceback
- Steps to reproduce

### 2. Share Benchmark Results

Help us understand performance across different hardware! Run:

```bash
python scripts/benchmark.py
```

Submit results via [GitHub Discussion](https://github.com/ChcharithOeun/torch-amd-setup/discussions) or Issue with:

- GPU model
- Driver version
- Python version
- Benchmark output (matmul, elementwise, MLP times)
- Speedup ratios (DirectML vs CPU)

### 3. Improve Documentation

- Fix typos or clarity issues in README.md, INSTALL.md
- Add new troubleshooting sections
- Share working use cases and examples
- Translate docs to other languages

### 4. Add Features

Potential improvements:
- ROCm support (Linux)
- Multi-GPU support utilities
- Performance profiling scripts
- Model-specific optimization guides
- Integration with popular frameworks (HuggingFace, TorchVision)
- Docker containerization

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/torch-amd-setup.git
   cd torch-amd-setup
   ```
3. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make changes, test thoroughly
5. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description of what changed and why"
   ```
6. Push and open a Pull Request

## Code Style

- Python: Follow [PEP 8](https://pep8.org/)
- Use docstrings for all functions and modules
- Add type hints where helpful
- Keep functions focused and testable
- Run `python -m py_compile scripts/*.py` to check syntax

## Testing

Before submitting a PR:

1. Test on your hardware:
   ```bash
   python scripts/verify_gpu.py
   python scripts/hello_gpu.py
   python scripts/benchmark.py
   ```

2. Verify your changes don't break existing scripts

3. Test edge cases (missing drivers, old Python versions, etc.)

## Commit Message Format

```
[type] Brief description

Longer explanation if needed. Include:
- Why this change was made
- What problem it solves
- Any breaking changes

Example:
[fix] Handle missing DirectML device gracefully
- Added try/except for device initialization
- Returns helpful error message instead of crash
- Fixes #42
```

Types: `[feature]`, `[fix]`, `[docs]`, `[test]`, `[refactor]`, `[perf]`

## PR Guidelines

- Link related issues: "Fixes #123"
- Keep PRs focused (one feature per PR)
- Include test results or benchmark outputs
- Update documentation if adding features
- Ensure CI passes (checks will run automatically)

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## Questions?

Open a [GitHub Discussion](https://github.com/ChharithOeun/torch-amd-setup/discussions) or reach out on the Issues page.

## Code of Conduct

This project is committed to being welcoming and inclusive. Be respectful, constructive, and collaborative.

## Support

If you found this project helpful and want to support more development:

[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20Me%20A%20Coffee-FFDD00?style=flat&logoColor=white)](https://buymeacoffee.com/chharith)

Happy contributing!
