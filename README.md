# Scribe

**Desktop OCR toolkit built on [scribe-frame](https://github.com/henryborner/scribe_frame).**

OCR · Document Parsing · Model Management — local GPU or remote cloud.

## Features

- PaddleOCR v6 engine (detection + recognition + layout + formula)
- PyQt6 dark-themed GUI with drag-drop, model download, hardware monitoring
- CLI: `scribe scan`, `scribe parse`, `scribe batch`
- Large-image auto-chunking (4000px+)
- Remote GPU offload to cloud server
- Plugin architecture — formatters/chunkers/commands auto-discover via entry_points
- Export: JSON, Markdown, plain text

## Quick start

```bash
pip install scribe
```

### GUI

```bash
scribe-gui
```

### CLI

```bash
scribe scan image.png -f text
scribe parse document.jpg -f markdown -o result.md
scribe batch *.png -o output/
```

### Install plugins

Any package registered under `scribe.providers`, `scribe.formatters`, `scribe.chunkers`, or `scribe.commands` is discovered automatically:

```bash
pip install my-custom-formatter
scribe scan image.png -f myfmt   # appears in dropdown without code changes
```

## Requirements

- Python >= 3.10
- NVIDIA GPU recommended (8GB+ VRAM for document parsing)
- CPU mode works for general OCR (slower)

## Architecture

```
scribe-gui / scribe CLI
        │
        ▼
    scribe_frame  ◄── plugin interfaces & registries
        │
   ┌────┼────┬──────────┐
   │    │    │          │
 Provider Formatter Chunker Commands
 (PaddleOCR) (JSON/MD/TXT) (default) (scan/parse/batch)
```

## License

MIT
