"""ChunkedPredictor — wraps an engine to handle large images transparently."""
from __future__ import annotations

from scribe_frame.chunker_registry import ChunkerRegistry


class ChunkedPredictor:
    """Middleware: chunk large images → predict each → merge results."""

    def __init__(self, engine, chunker=None):
        self._engine = engine
        self._chunker = chunker or ChunkerRegistry.get_default()

    def predict(self, input_path: str):
        """Full prediction with optional chunking."""
        import shutil
        from pathlib import Path

        chunks = self._chunker.chunk(input_path) if self._chunker else None

        # Always use _predict_single to avoid recursion
        predict_fn = getattr(self._engine, "_predict_single", self._engine.predict)

        if not chunks:
            # No chunking needed — normal path
            result = predict_fn(input_path)
            return result

        # Process each chunk
        all_results = []
        temp_dirs = set()
        for chunk in chunks:
            res = predict_fn(chunk.path)
            all_results.append(res)
            temp_dirs.add(Path(chunk.path).parent)

        # Merge results
        from scribe.results.base import OCRResult, StructureResult

        merged = all_results[0]
        if isinstance(merged, OCRResult):
            for i, res in enumerate(all_results[1:], 1):
                merged.merge_from(res, y_offset=chunks[i].y_offset)
        elif isinstance(merged, StructureResult):
            md_parts = []
            for res in all_results:
                if res.pages and "markdown" in res.pages[0].meta:
                    md_parts.append(res.pages[0].meta["markdown"])
            if md_parts:
                merged.pages[0].meta["markdown"] = "\n\n---\n\n".join(md_parts)
        else:
            # Generic: keep last
            pass

        # Cleanup
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

        return merged
