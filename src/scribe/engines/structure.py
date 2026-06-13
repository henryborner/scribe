"""Document parsing engine — wraps PPStructureV3 pipeline."""

from __future__ import annotations

import os
import sys
from typing import Iterator

from scribe.engines.base import BaseEngine
from scribe.providers.base import EnginePreset
from scribe.results.base import StructureResult, Page, Block, BBox, BlockType


def _fix_nvidia_dll_path() -> None:
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


class StructureEngine(BaseEngine):
    engine_type = "document_parsing"
    provider_name = "paddleocr"

    def __init__(self, preset: EnginePreset, device: str = "gpu", **kwargs):
        self._preset = preset
        self._config = {**preset.extra_config, **kwargs}
        self._pipeline = None

        _fix_nvidia_dll_path()
        from paddleocr import PPStructureV3

        table_model = preset.model_selections.get("table_structure_recognition")
        self._pipeline = PPStructureV3(
            layout_detection_model_name=preset.model_selections.get("layout_detection"),
            text_detection_model_name=preset.model_selections.get("text_detection"),
            text_recognition_model_name=preset.model_selections.get("text_recognition"),
            formula_recognition_model_name=preset.model_selections.get("formula_recognition"),
            wired_table_structure_recognition_model_name=table_model,
            wireless_table_structure_recognition_model_name=table_model,
            device=device,
            **self._config,
        )

    @property
    def model_info(self) -> dict:
        return {
            "preset": self._preset.id,
            **self._preset.model_selections,
        }

    def supported_formats(self) -> list[str]:
        return ["raw", "json", "markdown"]

    def predict(self, input_path: str | list[str]) -> StructureResult:
        from scribe.chunked_predictor import ChunkedPredictor

        chunker = ChunkedPredictor(self)

        paths = [input_path] if isinstance(input_path, str) else input_path

        md_parts = []
        raw = None
        for p in paths:
            res = chunker.predict(p)
            md_parts.append(res.pages[0].meta.get("markdown", "") if res.pages else "")
            if res.raw is not None:
                raw = res.raw

        md = "\n\n---\n\n".join(md_parts) if len(md_parts) > 1 else (md_parts[0] if md_parts else "")
        page = Page(index=0)
        page.meta["markdown"] = md
        return StructureResult(pages=[page], raw=raw)

    def _predict_single(self, input_path: str) -> StructureResult:
        """Predict a single image without chunking (used by ChunkedPredictor)."""
        raw = None
        md_text = ""
        for res in self._pipeline.predict(input_path):
            raw = res
            try:
                md_text = res.markdown.get("markdown_texts", "")
            except Exception:
                pass
        page = Page(index=0)
        page.meta["markdown"] = md_text
        return StructureResult(pages=[page], raw=raw)

    def predict_iter(self, input_path: str | list[str]) -> Iterator[StructureResult]:
        # For large images, predict_iter delegates to predict (chunked + merged)
        result = self.predict(input_path)
        yield result

    def close(self) -> None:
        self._pipeline = None
