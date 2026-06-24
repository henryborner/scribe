"""Scribe CLI — typer-based command line interface."""

from __future__ import annotations

import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.table import Table

# Fix nvidia DLL path before any paddle imports
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_base = os.path.join(_venv_root, "Lib", "site-packages", "nvidia")
if os.path.isdir(_nvidia_base):
    for _entry in os.listdir(_nvidia_base):
        _bin_dir = os.path.join(_nvidia_base, _entry, "bin")
        if os.path.isdir(_bin_dir) and _bin_dir not in os.environ.get("PATH", ""):
            try:
                os.add_dll_directory(_bin_dir)
            except AttributeError:
                pass
            os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

from scribe.hardware import detect
from scribe_frame.provider_registry import ProviderRegistry
from scribe_frame.formatter_registry import FormatterRegistry

app = typer.Typer(
    name="scribe",
    help="Pluggable OCR toolkit — OCR / Document Parsing / Model Management",
    no_args_is_help=True,
)
console = Console()
_provider = ProviderRegistry.get_default()

# ─────────────────────────── helpers ───────────────────────────

def _check_provider():
    if _provider is None or not _provider.check_availability():
        console.print("[red]❌ No OCR provider available. Install paddleocr[doc-parser][/red]")
        raise typer.Exit(1)


def _run_engine(engine_type: str, input_path: str, format_id: str, preset_id: str | None, device: str, output_path: str | None = None, quiet: bool = False):
    """Shared pipeline: select preset → create engine → predict → format."""
    _check_provider()
    hw = detect()

    # ── select preset ──
    if preset_id:
        presets = _provider.list_presets(engine_type)
        match = [p for p in presets if p.id == preset_id]
        if not match:
            console.print(f"[red]Preset '{preset_id}' not found. Available: {[p.id for p in presets]}[/red]")
            raise typer.Exit(1)
        preset = match[0]
    else:
        preset = _provider.recommend_preset(engine_type, hw)
        if not quiet:
            console.print(f"[dim]Auto-selected preset: {preset.name}[/dim]")

    # ── validate format ──
    engine = _provider.create_engine(engine_type, preset, device=device)
    if format_id not in engine.supported_formats():
        console.print(f"[red]Engine '{engine_type}' does not support format '{format_id}'. Supported: {engine.supported_formats()}[/red]")
        engine.close()
        raise typer.Exit(1)
    engine.close()

    # ── infer engine to get full model info ──
    if not quiet:
        console.print(f"[dim]Models: {preset.model_selections}[/dim]")

    # ── predict ──
    engine = _provider.create_engine(engine_type, preset, device=device)
    try:
        if not quiet:
            console.print("[yellow]Running inference...[/yellow]")
        result = engine.predict(input_path)
        formatter = FormatterRegistry.get(format_id)
        output = formatter.format(result)
        if not quiet:
            console.print(output)
        # --output: write to file, create if not exists
        if output_path:
            from pathlib import Path
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(output, encoding="utf-8")
            if not quiet:
                console.print(f"[green]Saved to {output_path}[/green]")
    finally:
        engine.close()


# ─────────────────────────── scan ───────────────────────────

@app.command()
def scan(
    input_path: str = typer.Argument(..., help="Image path or URL"),
    format: str = typer.Option(..., "--format", "-f", help="Output format: raw | json | text"),
    preset: str = typer.Option(None, "--preset", "-p", help="Preset (auto-selected if omitted)"),
    device: str = typer.Option("gpu", "--device", "-d", help="Device: gpu | cpu"),
    output: str = typer.Option(None, "--output", "-o", help="Save result to file path"),
):
    """General OCR — extract text from images."""
    _run_engine("general_ocr", input_path, format, preset, device, output)


# ─────────────────────────── parse ───────────────────────────

@app.command()
def parse(
    input_path: str = typer.Argument(..., help="Image/PDF path or URL"),
    format: str = typer.Option(..., "--format", "-f", help="Output format: raw | json | markdown"),
    preset: str = typer.Option(None, "--preset", "-p", help="Preset (auto-selected if omitted)"),
    device: str = typer.Option("gpu", "--device", "-d", help="Device: gpu | cpu"),
    output: str = typer.Option(None, "--output", "-o", help="Save result to file path"),
):
    """Document Parsing — extract text, tables, formulas as structured output."""
    _run_engine("document_parsing", input_path, format, preset, device, output)


# ─────────────────────────── batch ───────────────────────────

@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory of images/PDFs to process"),
    format: str = typer.Option("json", "--format", "-f", help="Output format"),
    preset: str = typer.Option(None, "--preset", "-p", help="Preset (auto-selected if omitted)"),
    device: str = typer.Option("gpu", "--device", "-d", help="Device: gpu | cpu"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory (default: input_dir/out)"),
    engine_type: str = typer.Option("general_ocr", "--engine", "-e", help="Engine: general_ocr | document_parsing"),
    print_results: bool = typer.Option(False, "--print", help="Print each result to console"),
):
    """Batch process all images in a directory. Results saved to files by default."""
    from pathlib import Path

    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        console.print(f"[red]Not a directory: {input_dir}[/red]")
        raise typer.Exit(1)

    out_dir = Path(output_dir) if output_dir else in_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf", ".webp"}
    files = sorted(
        f for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    if not files:
        console.print(f"[yellow]No images found in {input_dir}[/yellow]")
        return

    formatter = get_formatter(format)
    suffix = formatter.file_extension

    hw = detect()
    if not preset:
        p = _provider.recommend_preset(engine_type, hw)
        preset_id = p.id
        console.print(f"[dim]Auto preset: {p.name}[/dim]")
    else:
        preset_id = preset

    total = len(files)
    console.print(f"[bold]Batch: {total} file(s)[/bold] → {out_dir}/")

    for i, f in enumerate(files, 1):
        out_path = out_dir / f"{f.stem}{suffix}"
        status = f"[{i}/{total}] {f.name}"

        if out_path.exists():
            console.print(f"{status} [dim]skipped (exists)[/dim]")
            continue

        console.print(f"{status}", end=" ")
        try:
            _run_engine(engine_type, str(f), format, preset_id, device, str(out_path), quiet=not print_results)
            console.print(f"[green]✓[/green]")
        except typer.Exit:
            console.print(f"[red]✗[/red]")
        except Exception as e:
            console.print(f"[red]✗ {e}[/red]")

    console.print(f"\n[green]Done. {total} files → {out_dir}/[/green]")


# ─────────────────────────── models ───────────────────────────

@app.command()
def models(
    list_models: bool = typer.Option(False, "--list", "-l", help="List all available models"),
    presets: bool = typer.Option(False, "--presets", help="List all presets"),
    cache: bool = typer.Option(False, "--cache", help="Check model cache status"),
):
    """Model management — view models, presets, and cache status."""
    _check_provider()

    if list_models:
        table = Table(title="Available Models")
        table.add_column("Engine", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Name")
        table.add_column("Tier")
        for m in sorted(_provider.list_models(), key=lambda x: (x.engine_type, x.role, x.tier)):
            table.add_row(m.engine_type, m.role, m.name, m.tier)
        console.print(table)

    if presets:
        table = Table(title="Presets")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Engine")
        table.add_column("Min VRAM")
        for p in _provider.list_presets():
            table.add_row(p.id, p.name, p.engine_type, f"{p.recommended_vram_mb}MB")
        console.print(table)

    if cache:
        table = Table(title="Model Cache Status")
        table.add_column("Model", style="cyan")
        table.add_column("Cached")
        for m in sorted(_provider.list_models(), key=lambda x: (x.engine_type, x.role, x.name)):
            cached = "✅" if _provider.is_model_cached(m) else "⬜"
            table.add_row(m.name, cached)
        console.print(table)

    if not (list_models or presets or cache):
        hw = detect()
        console.print(f"[bold]Hardware:[/bold] {hw.gpu_name} ({hw.vram_total_mb}MB), recommended tier: {hw.recommended_tier}")
        console.print("[dim]Use --list / --presets / --cache for more info[/dim]")


# ─────────────────────────── init ───────────────────────────

@app.command()
def init():
    """Initialize — detect hardware and show recommended config."""
    hw = detect()
    console.print(f"[bold]Hardware[/bold]")
    console.print(f"  GPU: {hw.gpu_name} x {hw.gpu_count}")
    console.print(f"  VRAM: {hw.vram_total_mb}MB")
    console.print(f"  CPU: {hw.cpu_cores} cores / RAM: {hw.ram_total_mb}MB")
    console.print(f"  Recommended tier: [cyan]{hw.recommended_tier}[/cyan]  FP16: {'Yes' if hw.supports_fp16 else 'No'}")

    _check_provider()
    console.print(f"\n[bold]Recommended Presets[/bold]")
    for et in _provider.list_engine_types():
        p = _provider.recommend_preset(et, hw)
        console.print(f"  [{p.engine_type}] {p.name} ({', '.join(f'{k}={v}' for k, v in p.model_selections.items())})")


def main():
    app()

if __name__ == "__main__":
    main()
