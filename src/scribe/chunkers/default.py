"""Default image chunker — splits tall images into overlapping strips."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import tempfile

from scribe_frame.interfaces import BaseChunker, ChunkInfo

# Largest side length allowed (under PaddleOCR's 4000px internal limit)
MAX_CHUNK_SIDE = 4000
OVERLAP = 120


class DefaultChunker(BaseChunker):
    name = "default"

    def chunk(self, image_path: str) -> list[ChunkInfo] | None:
        img = None
        try:
            with open(image_path, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            pass

        if img is None:
            return None

        h, w = img.shape[:2]
        need_resize = w > MAX_CHUNK_SIDE
        chunk_h = MAX_CHUNK_SIDE
        chunk_w = MAX_CHUNK_SIDE if need_resize else w

        if h <= chunk_h:
            return None  # no chunking needed

        total = (h - 1) // (chunk_h - OVERLAP) + 1
        chunks = []
        temp_dir = Path(tempfile.mkdtemp(prefix="scribe_chunks_"))
        y = 0
        idx = 0
        while y < h:
            y2 = min(y + chunk_h, h)
            chunk_img = img[y:y2, :]
            if need_resize:
                chunk_img = cv2.resize(chunk_img, (chunk_w, y2 - y))
            chunk_path = temp_dir / f"chunk_{idx:04d}.png"
            cv2.imwrite(str(chunk_path), chunk_img)
            chunks.append(ChunkInfo(
                path=str(chunk_path),
                y_offset=y,
                index=idx,
                total=total,
            ))
            if y2 >= h:
                break
            y = y2 - OVERLAP
            idx += 1

        return chunks
