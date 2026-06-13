"""Quick smoke test for PaddleOCR provider."""
from scribe.providers.paddleocr import PaddleOCRProvider
from scribe.hardware import detect

p = PaddleOCRProvider()
print("可用:", p.check_availability())
print("引擎类型:", p.list_engine_types())

ocr_models = p.list_models("general_ocr")
print(f"\nOCR 模型: {len(ocr_models)} 个")
print(f"  检测: {[m.name for m in ocr_models if m.role == 'text_detection']}")
recs = [m.name for m in ocr_models if m.role == "text_recognition"]
print(f"  识别: {recs[:6]}... 共 {len(recs)} 个")

presets = p.list_presets()
print(f"\n预设: {len(presets)} 个")
for ps in presets:
    print(f"  [{ps.id}] {ps.name} — 需 {ps.recommended_vram_mb}MB 显存")

hw = detect()
rec = p.recommend_preset("general_ocr", hw)
print(f"\n🎯 基于 {hw.gpu_name} ({hw.vram_total_mb}MB) 推荐: {rec.name}")
