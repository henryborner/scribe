"""Formatter base class — output conversion plugin."""

from abc import ABC, abstractmethod


class BaseFormatter(ABC):
    """Converts an engine result into a specific output format."""

    @property
    @abstractmethod
    def format_id(self) -> str:
        """'raw' | 'json' | 'text' | 'markdown'."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """'.json' | '.txt' | '.md'."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name, e.g. 'JSON (structured)'."""

    @abstractmethod
    def supports(self, engine_type: str) -> bool:
        """Can this formatter handle the given engine_type?"""

    @abstractmethod
    def format(self, result, **options) -> str:
        """Convert result → formatted string."""

    def format_to_file(self, result, path: str, **options) -> None:
        """Write formatted output to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format(result, **options))
