"""Smoke test: create an OCREngine and run a prediction."""
import os, sys

# Fix: nvidia CUDA DLLs need to be on PATH (Windows-only issue)
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_bin = os.path.join(_venv_root, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
if os.path.isdir(_nvidia_bin):
    os.add_dll_directory(_nvidia_bin)
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

from scribe.providers.paddleocr import PaddleOCRProvider
from scribe.hardware import detect

p = PaddleOCRProvider()
hw = detect()

# ── Test 1: Create OCR engine ──
print("=" * 50)
print("Test 1: 创建 OCR 引擎")
preset = p.recommend_preset("general_ocr", hw)
print(f"  预设: {preset.name}")
print(f"  模型: det={preset.model_selections.get('text_detection')}, rec={preset.model_selections.get('text_recognition')}")

engine = p.create_engine("general_ocr", preset, device="gpu")
print(f"  引擎类型: {engine.engine_type}")
print(f"  支持格式: {engine.supported_formats()}")
print(f"  模型信息: {engine.model_info}")
print("  ✅ 创建成功")

# ── Test 2: Predict ──
print("\n" + "=" * 50)
print("Test 2: 执行预测")
img = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
print(f"  图片: {img}")
result = engine.predict(img)
print(f"  页数: {len(result.pages)}")
if result.pages:
    page = result.pages[0]
    print(f"  检测到 {len(page.blocks)} 个文本块")
    for b in page.blocks[:5]:
        txt = b.content[:30] + "..." if len(b.content) > 30 else b.content
        print(f"    [{b.bbox.x1},{b.bbox.y1}-{b.bbox.x2},{b.bbox.y2}] {txt}")
print("  ✅ 预测成功")

engine.close()
print("\n🎉 引擎插件就绪")
