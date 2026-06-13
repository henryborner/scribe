"""Markdown formatter — produces .md with optional LaTeX and tables."""

from scribe.formatters.base import BaseFormatter
from scribe.results.base import StructureResult


class MarkdownFormatter(BaseFormatter):
    format_id = "markdown"
    file_extension = ".md"
    label = "Markdown (formulas & tables)"

    def supports(self, engine_type: str) -> bool:
        return engine_type == "document_parsing"

    def format(self, result, **options) -> str:
        lines: list[str] = []
        for page in result.pages:
            # Use native markdown from engine if available
            if "markdown" in page.meta:
                lines.append(page.meta["markdown"])
            else:
                for block in page.blocks:
                    if block.type.value == "formula":
                        lines.append(f"$$\n{block.content}\n$$")
                    elif block.type.value == "title":
                        lines.append(f"## {block.content}")
                    else:
                        lines.append(block.content)
                lines.append("")
        return "\n\n".join(lines)
