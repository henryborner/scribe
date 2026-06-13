"""PyInstaller runtime hook — force-register bundled plugins.

When running as a PyInstaller bundle, importlib.metadata.entry_points()
can't discover plugins.  This hook imports them directly.
"""
from scribe_frame.provider_registry import ProviderRegistry
from scribe_frame.formatter_registry import FormatterRegistry
from scribe_frame.chunker_registry import ChunkerRegistry
from scribe_frame.command_registry import CommandRegistry

# ── Register all bundled plugins ──
try:
    from scribe.providers.paddleocr import PaddleOCRProvider
    ProviderRegistry.register(PaddleOCRProvider())
except Exception:
    pass

try:
    from scribe.formatters.json_fmt import JsonFormatter
    FormatterRegistry.register(JsonFormatter())
except Exception:
    pass

try:
    from scribe.formatters.text import TextFormatter
    FormatterRegistry.register(TextFormatter())
except Exception:
    pass

try:
    from scribe.formatters.markdown import MarkdownFormatter
    FormatterRegistry.register(MarkdownFormatter())
except Exception:
    pass

try:
    from scribe.chunkers.default import DefaultChunker
    ChunkerRegistry.register(DefaultChunker())
except Exception:
    pass

try:
    from scribe.commands.plugins import ScanCommand, ParseCommand, BatchCommand
    CommandRegistry.register(ScanCommand())
    CommandRegistry.register(ParseCommand())
    CommandRegistry.register(BatchCommand())
except Exception:
    pass
