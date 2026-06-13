"""Smoke test: full pipeline — Engine → Formatter."""
import os, sys

_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_bin = os.path.join(_venv_root, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
if os.path.isdir(_nvidia_bin):
    try:
        os.add_dll_directory(_nvidia_bin)
    except AttributeError:
        pass
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

from scribe.providers.paddleocr import PaddleOCRProvider
from scribe.hardware import detect
from scribe.formatters.registry import list_for_engine, get

p = PaddleOCRProvider()
hw = detect()

# ── OCR Engine → JSON / Text ──
print("=" * 50)
print("OCR 引擎 → JSON / Text")
preset = p.recommend_preset("general_ocr", hw)
engine = p.create_engine("general_ocr", preset, device="gpu")
result = engine.predict(
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)
engine.close()

for fmt in list_for_engine("general_ocr"):
    output = fmt.format(result)
    preview = output[:200].replace("\n", "\\n")
    print(f"\n  [{fmt.format_id}] {len(output)} 字符 → {preview}...")

print("\n🎉 Formatter 插件就绪")
