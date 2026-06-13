"""Profile actual VRAM usage for each OCR preset."""
import os, sys

_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_bin = os.path.join(_venv_root, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
if os.path.isdir(_nvidia_bin):
    try: os.add_dll_directory(_nvidia_bin)
    except AttributeError: pass
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

import paddle
from scribe.providers.paddleocr import PaddleOCRProvider

p = PaddleOCRProvider()
img = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"

def _vram_used_mb():
    """Return currently used GPU memory in MB."""
    return int(paddle.device.cuda.memory_allocated(0) / (1024 * 1024))

for preset in p.list_presets("general_ocr"):
    print(f"\n{'='*50}")
    print(f"[{preset.id}] {preset.name}")
    print(f"  det: {preset.model_selections.get('text_detection')}")
    print(f"  rec: {preset.model_selections.get('text_recognition')}")

    before = _vram_used_mb()
    engine = p.create_engine("general_ocr", preset, device="gpu")
    after_load = _vram_used_mb()
    print(f"  VRAM after load: {after_load - before:+d} MB  (total={after_load} MB)")

    engine.predict(img)
    after_predict = _vram_used_mb()
    engine.close()
    print(f"  VRAM after predict: {after_predict - before:+d} MB  (peak={after_predict} MB)")
    print(f"  Peak VRAM used: {after_predict} MB")
