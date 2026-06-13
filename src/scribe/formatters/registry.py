"""Formatter registry — map format_id to formatter class."""

from scribe.formatters.base import BaseFormatter
from scribe.formatters.raw import RawFormatter
from scribe.formatters.json_fmt import JsonFormatter
from scribe.formatters.text import TextFormatter
from scribe.formatters.markdown import MarkdownFormatter

_ALL: list[BaseFormatter] = [
    RawFormatter(),
    JsonFormatter(),
    TextFormatter(),
    MarkdownFormatter(),
]

_BY_ID: dict[str, BaseFormatter] = {f.format_id: f for f in _ALL}


def list_all() -> list[BaseFormatter]:
    return list(_ALL)


def get(format_id: str) -> BaseFormatter:
    if format_id not in _BY_ID:
        raise KeyError(f"Unknown format: {format_id}. Available: {list(_BY_ID)}")
    return _BY_ID[format_id]


def list_for_engine(engine_type: str) -> list[BaseFormatter]:
    return [f for f in _ALL if f.supports(engine_type)]
