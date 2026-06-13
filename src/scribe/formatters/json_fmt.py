"""JSON formatter — structured output with coordinates and confidence."""

import json as _json

from scribe.formatters.base import BaseFormatter
from scribe.results.base import OCRResult, StructureResult


class JsonFormatter(BaseFormatter):
    format_id = "json"
    file_extension = ".json"
    label = "JSON (structured)"

    def supports(self, engine_type: str) -> bool:
        return True

    def format(self, result, **options) -> str:
        data = {"engine_type": result.engine_type, "pages": []}
        for page in result.pages:
            page_data = {"index": page.index, "blocks": []}
            for block in page.blocks:
                b = {
                    "type": block.type.value,
                    "content": block.content,
                    "bbox": [block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2],
                    "confidence": round(block.confidence, 4),
                }
                if block.lines:
                    b["lines"] = [
                        {"text": ln.text, "bbox": [ln.bbox.x1, ln.bbox.y1, ln.bbox.x2, ln.bbox.y2]}
                        for ln in block.lines
                    ]
                page_data["blocks"].append(b)
            data["pages"].append(page_data)
        return _json.dumps(data, ensure_ascii=False, indent=2)
