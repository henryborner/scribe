"""Raw formatter — passes through the engine's native dict output."""

import json as _json

from scribe.formatters.base import BaseFormatter


class RawFormatter(BaseFormatter):
    format_id = "raw"
    file_extension = ".json"
    label = "Raw (engine output)"

    def supports(self, engine_type: str) -> bool:
        return True

    def format(self, result, **options) -> str:
        if hasattr(result, "raw") and result.raw is not None:
            return _json.dumps(result.raw, ensure_ascii=False, indent=2, default=str)
        return str(result)
