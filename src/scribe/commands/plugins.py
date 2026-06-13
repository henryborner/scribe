"""CLI command plugins — registered as scribe.commands entry_points."""
from __future__ import annotations

from scribe_frame.interfaces import BaseCommand
from scribe_frame.provider_registry import ProviderRegistry
from scribe_frame.formatter_registry import FormatterRegistry
from scribe.hardware import detect


def _get_provider():
    prov = ProviderRegistry.get_default()
    if prov is None or not prov.check_availability():
        raise RuntimeError("No OCR provider available")
    return prov


def _run_pipeline(engine_type: str, input_path: str, format_id: str,
                  preset_id: str | None, device: str):
    prov = _get_provider()
    hw = detect()

    if preset_id:
        presets = prov.list_presets(engine_type)
        match = [p for p in presets if p.id == preset_id]
        if not match:
            raise ValueError(f"Preset '{preset_id}' not found")
        preset = match[0]
    else:
        preset = prov.recommend_preset(engine_type, hw)

    engine = prov.create_engine(engine_type, preset, device=device)
    try:
        result = engine.predict(input_path)
        fmt = FormatterRegistry.get(format_id)
        if fmt is None:
            return result.all_text if hasattr(result, "all_text") else str(result)
        return fmt.format(result)
    finally:
        engine.close()


# ── scan ──

class ScanCommand(BaseCommand):
    name = "scan"
    help = "Run general OCR on an image"

    def register(self, parser) -> None:
        parser.add_argument("input", help="Image path")
        parser.add_argument("-p", "--preset", default=None, help="Preset ID")
        parser.add_argument("-d", "--device", default="gpu", help="gpu or cpu")
        parser.add_argument("-f", "--format", default="text", help="Output format")
        parser.add_argument("-o", "--output", default=None, help="Save to file")

    def run(self, args) -> int:
        try:
            text = _run_pipeline("general_ocr", args.input, args.format,
                                 args.preset, args.device)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved to {args.output}")
            else:
                print(text)
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1


# ── parse ──

class ParseCommand(BaseCommand):
    name = "parse"
    help = "Run document parsing on an image"

    def register(self, parser) -> None:
        parser.add_argument("input", help="Image path")
        parser.add_argument("-p", "--preset", default=None, help="Preset ID")
        parser.add_argument("-d", "--device", default="gpu", help="gpu or cpu")
        parser.add_argument("-f", "--format", default="markdown", help="Output format")
        parser.add_argument("-o", "--output", default=None, help="Save to file")

    def run(self, args) -> int:
        try:
            text = _run_pipeline("document_parsing", args.input, args.format,
                                 args.preset, args.device)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved to {args.output}")
            else:
                print(text)
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1


# ── batch ──

class BatchCommand(BaseCommand):
    name = "batch"
    help = "Batch process multiple images"

    def register(self, parser) -> None:
        parser.add_argument("inputs", nargs="+", help="Image paths")
        parser.add_argument("-e", "--engine", default="general_ocr",
                            help="general_ocr or document_parsing")
        parser.add_argument("-p", "--preset", default=None, help="Preset ID")
        parser.add_argument("-d", "--device", default="gpu", help="gpu or cpu")
        parser.add_argument("-f", "--format", default="text", help="Output format")
        parser.add_argument("-o", "--output-dir", default=None, help="Output directory")

    def run(self, args) -> int:
        import os
        for path in args.inputs:
            print(f"\n{'='*50}")
            print(f"Processing: {path}")
            try:
                text = _run_pipeline(args.engine, path, args.format,
                                     args.preset, args.device)
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(path))[0]
                    ext = FormatterRegistry.get(args.format)
                    ext = f".{ext.file_extension}" if ext else ".txt"
                    out_path = os.path.join(args.output_dir, base + ext)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"  → {out_path}")
                else:
                    print(text[:500])
            except Exception as e:
                print(f"  Error: {e}")
        return 0
