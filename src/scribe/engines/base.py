"""Engine base class — unified OCR interface."""

from abc import ABC, abstractmethod
from typing import Iterator


class BaseEngine(ABC):
    """Every OCR engine wraps an underlying pipeline behind this interface."""

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """'general_ocr' or 'document_parsing'."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """'paddleocr' etc."""

    @property
    @abstractmethod
    def model_info(self) -> dict:
        """Currently loaded model names and config."""

    @abstractmethod
    def predict(self, input_path: str | list[str]):
        """Run OCR on image(s) or PDF, return an *EngineResult*."""
        ...

    @abstractmethod
    def predict_iter(self, input_path: str | list[str]) -> Iterator:
        """Streaming variant — yield results per page / image."""
        ...

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return format IDs this engine can produce natively.
        e.g. ['raw', 'json', 'markdown']."""

    @abstractmethod
    def close(self) -> None:
        """Release GPU resources."""
