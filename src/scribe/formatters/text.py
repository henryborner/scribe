"""Plain-text formatter with smart word-wrapping for OCR results."""

from scribe.formatters.base import BaseFormatter
from scribe.results.base import OCRResult, StructureResult


class TextFormatter(BaseFormatter):
    format_id = "text"
    file_extension = ".txt"
    label = "Plain Text (smart wrap)"

    def supports(self, engine_type: str) -> bool:
        return True

    def format(self, result, **options) -> str:
        if isinstance(result, OCRResult):
            return result.all_text
        # StructureResult → fallback: dump block contents
        lines = []
        for page in result.pages:
            for block in page.blocks:
                if block.type.value == "table":
                    lines.append(f"[Table: {block.content[:80]}...]" if len(block.content) > 80 else f"[Table: {block.content}]")
                elif block.type.value == "formula":
                    lines.append(f"$$ {block.content} $$")
                else:
                    lines.append(block.content)
            lines.append("")
        return "\n".join(lines)
