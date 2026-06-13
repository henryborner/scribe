"""Result base class and format enumeration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    SEAL = "seal"
    CHART = "chart"


@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int


@dataclass
class TextLine:
    text: str
    bbox: BBox
    confidence: float = 0.0


@dataclass
class Block:
    type: BlockType
    content: str                    # text / LaTeX / HTML table
    bbox: BBox
    confidence: float = 0.0
    lines: list[TextLine] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class OCRResult:
    """Flat text-line result from general OCR engine."""
    engine_type: str = "general_ocr"
    pages: list["Page"] = field(default_factory=list)
    raw: Optional[dict] = None      # engine-native output for debugging

    @property
    def all_text(self) -> str:
        return "\n".join(
            line.text for page in self.pages for block in page.blocks for line in block.lines
        )

    def merge_from(self, other: "OCRResult", y_offset: int = 0) -> None:
        """Merge another OCRResult into this one, applying y_offset to all boxes."""
        for page in other.pages:
            for block in page.blocks:
                block.bbox.y1 += y_offset
                block.bbox.y2 += y_offset
                for line in block.lines:
                    line.bbox.y1 += y_offset
                    line.bbox.y2 += y_offset
        if self.pages:
            self.pages[0].blocks.extend(other.pages[0].blocks if other.pages else [])
        else:
            self.pages = other.pages
        if other.raw is not None:
            if isinstance(self.raw, list):
                self.raw.append(other.raw if not isinstance(other.raw, list) else other.raw)
            elif isinstance(other.raw, list):
                self.raw = [self.raw] + other.raw if self.raw is not None else other.raw
            else:
                self.raw = [self.raw, other.raw] if self.raw is not None else [other.raw]


@dataclass
class StructureResult:
    """Structured layout result from document parsing engine."""
    engine_type: str = "document_parsing"
    pages: list["Page"] = field(default_factory=list)
    raw: Optional[dict] = None

    @property
    def all_text(self) -> str:
        return "\n\n".join(block.content for page in self.pages for block in page.blocks)


@dataclass
class Page:
    index: int = 0
    blocks: list[Block] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
