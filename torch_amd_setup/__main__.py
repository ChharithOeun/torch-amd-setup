"""python -m torch_amd_setup  →  run diagnostics CLI"""
import logging
from .detect import device_info, get_install_guide

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

print("\n── torch-amd-setup diagnostics ──────────────────────────────")
info = device_info()
for k, v in info.items():
    print(f"  {k:<25} {v}")
print()
print("── Install guide ─────────────────────────────────────────────")
print(get_install_guide())
