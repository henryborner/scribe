"""Provider base class — OCR backend abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    """Metadata for a single model."""
    id: str                     # "pp_ocr_v6_mobile_det"
    name: str                   # "PP-OCRv6_mobile_det"
    provider: str               # "paddleocr"
    engine_type: str            # "general_ocr" | "document_parsing"
    role: str                   # "detection" | "recognition" | "layout" | "formula" | "table"
    display_name: str           # "PP-OCRv6 Mobile Detection"
    tier: str = "mobile"        # "tiny" | "mobile" | "medium" | "server"
    vram_estimate_mb: int = 0
    download_size_mb: int = 0
    description: str = ""
    version: str = ""
    languages: list[str] = field(default_factory=lambda: ["zh", "en"])
    requires: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    required_provider: str = ""


@dataclass
class EnginePreset:
    """A ready-to-use combination of models for an engine."""
    id: str
    name: str
    engine_type: str
    recommended_vram_mb: int
    model_selections: dict          # {"detection": "PP-OCRv6_mobile_det", ...}
    extra_config: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """Each OCR backend (PaddleOCR, Tesseract, ...) implements this."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier, e.g. 'paddleocr'."""

    @abstractmethod
    def list_engine_types(self) -> list[str]:
        """Supported engine types like ['general_ocr', 'document_parsing']."""

    @abstractmethod
    def list_models(self, engine_type: Optional[str] = None) -> list[ModelSpec]:
        """All models this provider offers."""

    @abstractmethod
    def list_presets(self, engine_type: Optional[str] = None) -> list[EnginePreset]:
        """Preset combinations for quick selection."""

    @abstractmethod
    def check_availability(self) -> bool:
        """Can this provider run in the current environment?"""

    @abstractmethod
    def create_engine(self, engine_type: str, preset: EnginePreset, **config):
        """Instantiate a ready-to-use engine from a preset."""

    @abstractmethod
    def is_model_cached(self, model: ModelSpec) -> bool:
        """Check if the model is downloaded locally."""

    @abstractmethod
    def download_model(self, model: ModelSpec, on_progress=None) -> bool:
        """Download a model to local cache."""
