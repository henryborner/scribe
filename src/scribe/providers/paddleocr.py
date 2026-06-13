"""PaddleOCR provider — discovers models from PaddleX configs and wraps engines."""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Optional

from scribe.providers.base import BaseProvider, ModelSpec, EnginePreset
from scribe.hardware import HardwareInfo

# ── static registry built from PaddleX config YAML files ──────────────

_PADDLEX_CONFIG_ROOT = None


def _get_paddlex_config_root() -> str:
    global _PADDLEX_CONFIG_ROOT
    if _PADDLEX_CONFIG_ROOT is None:
        import paddlex
        _PADDLEX_CONFIG_ROOT = os.path.join(
            os.path.dirname(paddlex.__file__), "configs", "modules"
        )
    return _PADDLEX_CONFIG_ROOT


def _scan_models() -> list[ModelSpec]:
    """Walk PaddleX configs/modules/ and extract every OCR-relevant ModelSpec."""
    models: list[ModelSpec] = []
    root = _get_paddlex_config_root()
    for folder in sorted(os.listdir(root)):
        if folder not in _VALID_ROLES:
            continue
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fn in sorted(os.listdir(folder_path)):
            if not fn.endswith(".yaml"):
                continue
            model_name = fn.replace(".yaml", "")
            model_path = os.path.join(folder_path, fn)
            try:
                with open(model_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}

            display = cfg.get("display_name", model_name)
            models.append(ModelSpec(
                id=f"paddleocr.{folder}.{model_name}",
                name=model_name,
                provider="paddleocr",
                engine_type="general_ocr" if folder in _OCR_ROLES else "document_parsing",
                role=folder,
                display_name=display,
                tier=_infer_tier(model_name),
                description=cfg.get("description", ""),
                languages=cfg.get("languages", ["zh", "en"]),
            ))
    return models


_OCR_ROLES = {
    "text_detection", "text_recognition", "doc_text_orientation",
    "textline_orientation", "image_unwarping",
}

_STRUCTURE_ROLES = {
    "layout_detection", "layout_analysis", "formula_recognition",
    "table_structure_recognition", "table_cells_detection",
    "seal_text_detection", "chart_parsing", "table_classification",
}

_VALID_ROLES = _OCR_ROLES | _STRUCTURE_ROLES


def _infer_tier(name: str) -> str:
    name_lower = name.lower()
    if "server" in name_lower or "plus" in name_lower or "large" in name_lower or "-l" in name_lower:
        return "server"
    if "medium" in name_lower or "base" in name_lower or "-m" in name_lower:
        return "medium"
    if "tiny" in name_lower or "small" in name_lower or "-s" in name_lower:
        return "mobile"
    if "mobile" in name_lower:
        return "mobile"
    return "medium"


# ── presets ───────────────────────────────────────────────────────────

_PRESETS: list[EnginePreset] = [
    EnginePreset(
        id="ocr_fast",
        name="OCR · Fast",
        engine_type="general_ocr",
        recommended_vram_mb=1500,
        model_selections={
            "text_detection": "PP-OCRv6_tiny_det",
            "text_recognition": "PP-OCRv6_tiny_rec",
        },
    ),
    EnginePreset(
        id="ocr_balanced",
        name="OCR · Balanced",
        engine_type="general_ocr",
        recommended_vram_mb=2000,
        model_selections={
            "text_detection": "PP-OCRv6_small_det",
            "text_recognition": "PP-OCRv6_small_rec",
        },
    ),
    EnginePreset(
        id="ocr_accurate",
        name="OCR · Accurate",
        engine_type="general_ocr",
        recommended_vram_mb=3000,
        model_selections={
            "text_detection": "PP-OCRv6_medium_det",
            "text_recognition": "PP-OCRv6_medium_rec",
        },
    ),
    EnginePreset(
        id="structure_mobile",
        name="Doc Parse · 10GB VRAM",
        engine_type="document_parsing",
        recommended_vram_mb=10000,
        model_selections={
            "layout_detection": "PP-DocBlockLayout",
            "text_detection": "PP-OCRv6_mobile_det",
            "text_recognition": "PP-OCRv6_mobile_rec",
            "formula_recognition": "PP-FormulaNet-S",
            "table_structure_recognition": "SLANet",
        },
        extra_config={
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "precision": "fp16",
        },
    ),
    EnginePreset(
        id="structure_server",
        name="Doc Parse · 16GB+ VRAM",
        engine_type="document_parsing",
        recommended_vram_mb=16000,
        model_selections={
            "layout_detection": "PP-DocLayout_plus-L",
            "text_detection": "PP-OCRv6_medium_det",
            "text_recognition": "PP-OCRv6_medium_rec",
            "formula_recognition": "PP-FormulaNet-L",
            "table_structure_recognition": "SLANet_plus",
        },
    ),
]


# ── provider implementation ───────────────────────────────────────────

CATALOG_URL = "https://henryborner.top/api/ocr/catalog.json"
CACHE_DIR = os.path.expanduser("~/.scribe")
CACHE_TTL_SECONDS = 86400  # 24 hours


class PaddleOCRProvider(BaseProvider):
    name = "paddleocr"

    def __init__(self):
        self._models: Optional[list[ModelSpec]] = None
        self._catalog_source: str = ""

    @property
    def catalog_source(self) -> str:
        """Where the current model list came from: 'remote', 'cache', 'local'."""
        return self._catalog_source

    # ── remote catalog ─────────────────────────────────────────

    @staticmethod
    def _fetch_remote_catalog() -> tuple[list[ModelSpec], str]:
        """Fetch model catalog from server. Returns (models, source_label)."""
        import json, urllib.request

        try:
            req = urllib.request.Request(CATALOG_URL)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return [], ""

        models: list[ModelSpec] = []
        for m in data.get("models", []):
            models.append(ModelSpec(
                id=m["id"],
                name=m["name"],
                provider="paddleocr",
                engine_type=m["engine"],
                role=m["role"],
                display_name=m.get("display_name", m["name"]),
                tier=m.get("tier", "medium"),
                description=m.get("description", ""),
                languages=m.get("languages", ["zh", "en"]),
            ))
        return models, "remote"

    @staticmethod
    def _load_catalog_cache() -> tuple[list[ModelSpec], str] | None:
        """Load cached catalog from disk if not expired."""
        import json, time

        cache_file = os.path.join(CACHE_DIR, "catalog_cache.json")
        if not os.path.exists(cache_file):
            return None

        try:
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime > CACHE_TTL_SECONDS:
                return None  # expired
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        models: list[ModelSpec] = []
        for m in data.get("models", []):
            models.append(ModelSpec(
                id=m["id"],
                name=m["name"],
                provider="paddleocr",
                engine_type=m["engine"],
                role=m["role"],
                display_name=m.get("display_name", m["name"]),
                tier=m.get("tier", "medium"),
                description=m.get("description", ""),
                languages=m.get("languages", ["zh", "en"]),
            ))
        return models, "cache"

    @staticmethod
    def _save_catalog_cache(models: list[ModelSpec]) -> None:
        """Persist catalog to local cache."""
        import json

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, "catalog_cache.json")
        data = {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "engine": m.engine_type,
                    "role": m.role,
                    "display_name": m.display_name,
                    "tier": m.tier,
                    "description": m.description,
                    "languages": m.languages,
                }
                for m in models
            ]
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def refresh_remote_catalog(self) -> str:
        """Force re-fetch from server. Returns source label."""
        self._models = None
        self._catalog_source = ""
        self._load_models()
        return self._catalog_source

    # ── model loading ──────────────────────────────────────────

    def _load_models(self) -> list[ModelSpec]:
        if self._models is not None:
            return self._models

        # 1) Try remote
        models, source = self._fetch_remote_catalog()
        if models:
            self._models = models
            self._catalog_source = source
            self._save_catalog_cache(models)
            return self._models

        # 2) Try local cache
        cached = self._load_catalog_cache()
        if cached is not None:
            self._models = cached[0]
            self._catalog_source = cached[1]
            return self._models

        # 3) Fallback: local PaddleX scan
        self._models = _scan_models()
        self._catalog_source = "local"
        return self._models

    def list_engine_types(self) -> list[str]:
        return ["general_ocr", "document_parsing"]

    def list_models(self, engine_type: Optional[str] = None) -> list[ModelSpec]:
        all_models = self._load_models()
        if engine_type is None:
            return all_models
        return [m for m in all_models if m.engine_type == engine_type]

    def list_presets(self, engine_type: Optional[str] = None) -> list[EnginePreset]:
        if engine_type is None:
            return _PRESETS
        return [p for p in _PRESETS if p.engine_type == engine_type]

    def check_availability(self) -> bool:
        try:
            import paddleocr  # noqa
            return True
        except ImportError:
            return False

    def create_engine(self, engine_type: str, preset: EnginePreset, **config):
        if engine_type == "general_ocr":
            from scribe.engines.ocr import OCREngine
            return OCREngine(preset=preset, **config)
        elif engine_type == "document_parsing":
            from scribe.engines.structure import StructureEngine
            return StructureEngine(preset=preset, **config)
        raise ValueError(f"Unknown engine_type: {engine_type}")

    def is_model_cached(self, model: ModelSpec) -> bool:
        cache_dir = os.path.expanduser(
            f"~/.paddlex/official_models/{model.name}"
        )
        return os.path.isdir(cache_dir) and any(
            f.endswith((".pdiparams", ".pdmodel", ".json"))
            for f in os.listdir(cache_dir)
        ) if os.path.isdir(cache_dir) else False

    def download_model(self, model: ModelSpec, on_progress=None) -> bool:
        """Trigger download by instantiating the engine briefly.
        PaddleX handles actual download + caching automatically."""
        try:
            preset = EnginePreset(
                id="__download__",
                name="Download",
                engine_type=model.engine_type,
                recommended_vram_mb=0,
                model_selections={model.role: model.name},
            )
            engine = self.create_engine(model.engine_type, preset)
            engine.close()
            return self.is_model_cached(model)
        except Exception:
            return False

    def recommend_preset(
        self,
        engine_type: str,
        hw: Optional[HardwareInfo] = None,
    ) -> EnginePreset:
        """Pick the best preset based on hardware."""
        presets = self.list_presets(engine_type)
        if not presets:
            raise ValueError(f"No presets for {engine_type}")

        if hw is None:
            return presets[0]

        # filter by VRAM
        viable = [p for p in presets if p.recommended_vram_mb <= hw.vram_total_mb]
        if not viable:
            return presets[0]  # fallback: smallest
        return viable[-1]      # pick the most capable that fits
