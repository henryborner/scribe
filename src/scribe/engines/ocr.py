"""General OCR engine — wraps PaddleOCR pipeline."""

from __future__ import annotations

import os
import sys
from typing import Iterator

from scribe.engines.base import BaseEngine
from scribe.providers.base import EnginePreset
from scribe.results.base import OCRResult, Page, Block, TextLine, BBox, BlockType


def _fix_nvidia_dll_path() -> None:
    """Ensure nvidia CUDA DLLs are findable on Windows."""
    if sys.platform != "win32":
        return
    _venv = os.path.dirname(os.path.dirname(sys.executable))
    _nvidia_bin = os.path.join(_venv, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
    if os.path.isdir(_nvidia_bin) and _nvidia_bin not in os.environ.get("PATH", ""):
        try:
            os.add_dll_directory(_nvidia_bin)
        except AttributeError:
            pass
        os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")


class OCREngine(BaseEngine):
    engine_type = "general_ocr"
    provider_name = "paddleocr"

    def __init__(self, preset: EnginePreset, device: str = "gpu", **kwargs):
        self._preset = preset
        self._config = {**preset.extra_config, **kwargs}
        self._pipeline = None

        _fix_nvidia_dll_path()
        from paddleocr import PaddleOCR
        det = preset.model_selections.get("text_detection")
        rec = preset.model_selections.get("text_recognition")

        self._pipeline = PaddleOCR(
            text_detection_model_name=det,
            text_recognition_model_name=rec,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            **self._config,
        )

    @property
    def model_info(self) -> dict:
        return {
            "preset": self._preset.id,
            "detection": self._preset.model_selections.get("text_detection"),
            "recognition": self._preset.model_selections.get("text_recognition"),
        }

    def supported_formats(self) -> list[str]:
        return ["raw", "json", "text"]

    def predict(self, input_path: str | list[str]) -> OCRResult:
        from scribe.chunked_predictor import ChunkedPredictor

        chunker = ChunkedPredictor(self)

        paths = [input_path] if isinstance(input_path, str) else input_path

        if len(paths) == 1:
            return chunker.predict(paths[0])

        # Multiple images — chunk each and merge
        all_blocks: list[Block] = []
        all_raw: list = []
        for p in paths:
            res = chunker.predict(p)
            for page in res.pages:
                all_blocks.extend(page.blocks)
            if res.raw is not None:
                all_raw.append(res.raw)

        return OCRResult(pages=[Page(index=0, blocks=all_blocks)], raw=all_raw)

    def _predict_single(self, input_path: str) -> OCRResult:
        """Predict a single image without chunking (used by ChunkedPredictor)."""
        blocks = []
        raw = None
        for res in self._pipeline.predict(input_path):
            raw = res
            for text, box in zip(res.get("rec_texts", []), res.get("rec_boxes", [])):
                if not text or not text.strip():
                    continue
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                bbox = BBox(x1, y1, x2, y2)
                blocks.append(Block(
                    type=BlockType.TEXT,
                    content=text,
                    bbox=bbox,
                    lines=[TextLine(text=text, bbox=bbox)],
                ))
        return OCRResult(pages=[Page(index=0, blocks=blocks)], raw=raw)

    def predict_iter(self, input_path: str | list[str]) -> Iterator[OCRResult]:
        for res in self._pipeline.predict_iter(input_path):
            blocks = []
            for text, box in zip(res.get("rec_texts", []), res.get("rec_boxes", [])):
                if not text or not text.strip():
                    continue
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                bbox = BBox(x1, y1, x2, y2)
                blocks.append(Block(
                    type=BlockType.TEXT,
                    content=text,
                    bbox=bbox,
                    lines=[TextLine(text=text, bbox=bbox)],
                ))
            yield OCRResult(pages=[Page(index=0, blocks=blocks)], raw=res)

    def close(self) -> None:
        self._pipeline = None
