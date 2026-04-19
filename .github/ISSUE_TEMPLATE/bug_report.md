---
name: Bug report
about: Report a problem with torch-amd-setup
title: "[BUG] "
labels: bug
assignees: ''

---

## Description

A clear and concise description of what the bug is.

## Environment

- **GPU Model**: (e.g., AMD Radeon RX 6700 XT, Intel Arc A770, Nvidia RTX 4060)
- **GPU Driver Version**: (e.g., from AMD Radeon Software or Intel Arc Control)
- **Operating System**: Windows 10/11, version
- **Python Version**: (from `python --version`)
- **torch-directml Version**: (from `python -c "import torch_directml; print(torch_directml.__version__)"`)
- **PyTorch Version**: (from `python -c "import torch; print(torch.__version__)"`)

## Steps to Reproduce

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

Describe what you expected to happen.

## Actual Behavior

Describe what actually happened.

## Error Output

```
Paste the full error message, traceback, or output here
```

## Additional Context

Add any other context about the problem here:
- Recent changes to your system
- Updated drivers
- Other PyTorch installations
- Relevant code snippets

## Verification Steps Completed

- [ ] Ran `python scripts/verify_gpu.py` successfully
- [ ] Ran `python scripts/hello_gpu.py` successfully
- [ ] GPU drivers are up to date
- [ ] No conflicting PyTorch installations (CUDA version)
- [ ] Python 3.10 or higher
