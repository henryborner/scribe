"""Scribe GUI — PyQt6 desktop interface with tabs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Fix nvidia DLL path before anything else ──
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_bin = os.path.join(_venv_root, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
if os.path.isdir(_nvidia_bin) and _nvidia_bin not in os.environ.get("PATH", ""):
    try:
        os.add_dll_directory(_nvidia_bin)
    except AttributeError:
        pass
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QComboBox, QLabel, QFileDialog,
    QSplitter, QMessageBox, QProgressBar,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QFrame, QCheckBox, QLineEdit, QGroupBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont

from scribe_frame.provider_registry import ProviderRegistry
from scribe_frame.formatter_registry import FormatterRegistry
from scribe_frame.interfaces import EnginePreset
from scribe_frame.ui.theme import DARK_STYLE
from scribe_frame.ui.widgets import (
    DarkButton, PrimaryButton, OutlineButton,
    DropZone, ResultPanel, HardwarePanel,
)
from scribe.hardware import detect
from scribe.formatters.raw import RawFormatter

# Built-in fallback: Raw
FormatterRegistry.register(RawFormatter())

# ─────────────────────────── worker threads ───────────────────────────

class OCRWorker(QThread):
    """Run OCR in background so GUI stays responsive."""
    started = pyqtSignal()
    finished = pyqtSignal(str)      # formatted output
    error = pyqtSignal(str)

    def __init__(self, engine_type: str, input_path: str, preset_id: str, format_id: str, device: str, preset=None):
        super().__init__()
        self.engine_type = engine_type
        self.input_path = input_path
        self.preset_id = preset_id
        self.format_id = format_id
        self.device = device
        self._direct_preset = preset  # bypass preset lookup

    def run(self):
        self.started.emit()
        try:
            provider = ProviderRegistry.get_default()
            if self._direct_preset:
                preset = self._direct_preset
            else:
                presets = provider.list_presets(self.engine_type)
                preset = next(p for p in presets if p.id == self.preset_id)
            engine = provider.create_engine(self.engine_type, preset, device=self.device)
            try:
                result = engine.predict(self.input_path)
                fmt = FormatterRegistry.get(self.format_id)
                output = fmt.format(result)
                self.finished.emit(output)
            finally:
                engine.close()
        except Exception as e:
            self.error.emit(str(e))


class DownloadWorker(QThread):
    """Download models in background, capturing tqdm progress from stderr."""
    log = pyqtSignal(str)                # log message
    progress = pyqtSignal(int, int)      # current_bytes, total_bytes
    model_start = pyqtSignal(str)        # model name starting
    model_done = pyqtSignal(str, bool)   # model name, success
    all_done = pyqtSignal(int, int)      # ok count, fail count

    def __init__(self, models: list):
        super().__init__()
        self.models = models

    def run(self):
        import datetime, io, re, sys, threading

        # Regex to strip ANSI escape codes (tqdm cursor control chars)
        _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

        def _clean_line(s: str) -> str:
            return _ANSI_RE.sub("", s)

        provider = ProviderRegistry.get_default()
        ok_count = 0
        fail_count = 0
        self.log.emit(f"{'='*50}")
        self.log.emit(f"  Download session — {len(self.models)} model(s)")
        self.log.emit(f"{'='*50}")

        # Regex to parse tqdm lines like:
        # "Downloading [inference.pdiparams]: 100%|...| 1.64M/1.64M [00:01<00:00, 1.86MB/s]"
        _TQDM_RE = re.compile(
            r"(\d+)%\|[^|]+\|\s*([\d.]+)([KMGT]?)\s*/\s*([\d.]+)([KMGT]?)"
        )
        _SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

        def _parse_size(val: str, unit: str) -> int:
            return int(float(val) * _SIZE_UNITS.get(unit, 1))

        for m in self.models:
            self.model_start.emit(m.name)
            self.log.emit(
                f"[{datetime.datetime.now():%H:%M:%S}] ⬇ {m.name} "
                f"({m.role}/{m.tier})"
            )

            # Redirect stderr to capture tqdm output
            old_stderr = sys.stderr
            capture = io.StringIO()
            sys.stderr = capture

            # Poll captured stderr in a daemon thread
            stop_poll = threading.Event()
            seen_pos = [0]  # mutable for closure

            def _poll_stderr():
                while not stop_poll.is_set():
                    stop_poll.wait(0.3)
                    text = capture.getvalue()
                    new_text = text[seen_pos[0]:]
                    seen_pos[0] = len(text)
                    for line in new_text.splitlines():
                        stripped = _clean_line(line).strip()
                        if not stripped:
                            continue
                        m2 = _TQDM_RE.search(stripped)
                        if m2:
                            cur = _parse_size(m2.group(2), m2.group(3))
                            tot = _parse_size(m2.group(4), m2.group(5))
                            if tot > 0:
                                self.progress.emit(cur, tot)
                        elif any(kw in stripped.lower() for kw in
                                 ("downloading", "model file", "error", "warning",
                                  "finish", "saved", "cache")):
                            self.log.emit(f"  {stripped[:150]}")

            poller = threading.Thread(target=_poll_stderr, daemon=True)
            poller.start()

            try:
                ok = provider.download_model(m)
            except Exception as e:
                ok = False
                self.log.emit(
                    f"[{datetime.datetime.now():%H:%M:%S}] ❌ {m.name} — {e}"
                )
            finally:
                stop_poll.set()
                poller.join(timeout=1)
                sys.stderr = old_stderr

            if ok:
                self.log.emit(
                    f"[{datetime.datetime.now():%H:%M:%S}] ✅ {m.name}"
                )
                self.model_done.emit(m.name, True)
                ok_count += 1
            else:
                self.log.emit(
                    f"[{datetime.datetime.now():%H:%M:%S}] ⚠ {m.name} failed"
                )
                self.model_done.emit(m.name, False)
                fail_count += 1

        self.log.emit(f"{'='*50}")
        self.log.emit(f"  Done: {ok_count} OK, {fail_count} failed")
        self.log.emit(f"{'='*50}")
        self.all_done.emit(ok_count, fail_count)


class RemoteWorker(QThread):
    """Send image to remote OCR server."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url: str, image_path: str):
        super().__init__()
        self.url = url.rstrip("/")
        self.image_path = image_path

    def run(self):
        import base64, requests
        try:
            with open(self.image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            r = requests.post(
                self.url,
                json={"file": b64, "fileType": 1},
                timeout=300,
            )
            if r.status_code != 200:
                self.error.emit(f"Server returned {r.status_code}: {r.text[:300]}")
                return
            data = r.json()
            # Extract markdown text from PaddleOCR-VL response
            results = data.get("result", {}).get("layoutParsingResults", [])
            if results:
                md = results[0].get("markdown", {}).get("text", "")
                self.finished.emit(md)
            else:
                self.finished.emit(str(data)[:5000])
        except Exception as e:
            self.error.emit(str(e))


from scribe_frame.ui.theme import DARK_STYLE
from scribe_frame.ui.widgets import (
    DarkButton, PrimaryButton, OutlineButton,
    DropZone, ResultPanel, HardwarePanel,
)


# ─────────────────────────── main window ───────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scribe — OCR Toolkit")
        self.setMinimumSize(1100, 700)

        self._provider = ProviderRegistry.get_default() or None
        self._input_path: str | None = None
        self._worker: OCRWorker | None = None
        self._download_worker: DownloadWorker | None = None
        self._remote_worker: RemoteWorker | None = None
        self._viable_presets: set[str] = set()
        self._model_items: dict[str, QTreeWidgetItem] = {}

        self._setup_ui()
        self._populate_presets()

    # ── UI layout ─────────────────────────────────────────────────

    def _setup_ui(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._setup_ocr_tab()
        self._setup_hardware_tab()
        self._setup_models_tab()
        self._setup_downloads_tab()

        self._tabs.currentChanged.connect(self._on_tab_changed)
        self.setStyleSheet(DARK_STYLE)
        self.setAcceptDrops(True)

    # ──────────────── OCR tab ────────────────

    def _setup_ocr_tab(self):
        tab = QWidget()
        root = QHBoxLayout(tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── LEFT panel ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Drop zone
        self._drop_zone = DropZone("Drop Image / PDF Here")
        self._drop_zone.file_selected.connect(self._set_input)
        left_layout.addWidget(self._drop_zone)

        # Engine selector
        left_layout.addWidget(QLabel("Engine"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItem("General OCR", "general_ocr")
        self._engine_combo.addItem("Document Parsing", "document_parsing")
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        left_layout.addWidget(self._engine_combo)

        # Preset selector
        left_layout.addWidget(QLabel("Preset"))
        self._preset_combo = QComboBox()
        self._preset_combo.currentIndexChanged.connect(self._check_run_enabled)
        left_layout.addWidget(self._preset_combo)

        # Format selector
        left_layout.addWidget(QLabel("Output Format"))
        self._format_combo = QComboBox()
        left_layout.addWidget(self._format_combo)

        # Run button
        self._run_btn = PrimaryButton("Run OCR")
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setEnabled(False)
        left_layout.addWidget(self._run_btn)

        # Remote server toggle
        remote_layout = QHBoxLayout()
        self._remote_cb = QCheckBox("☁ Remote")
        self._remote_cb.setToolTip("Send to cloud server when VRAM insufficient")
        self._remote_cb.setStyleSheet("color: #888;")
        self._remote_cb.toggled.connect(self._on_remote_toggled)
        remote_layout.addWidget(self._remote_cb)
        self._remote_url = QLineEdit()
        self._remote_url.setPlaceholderText("http://server:8080/layout-parsing")
        self._remote_url.setStyleSheet("background:#1e1e2e;color:#888;border:1px solid #333;border-radius:4px;padding:2px 6px;")
        self._remote_url.setVisible(False)
        self._remote_url.setMaximumHeight(24)
        remote_layout.addWidget(self._remote_url)
        left_layout.addLayout(remote_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)

        # Advanced mode button
        self._adv_btn = OutlineButton("⚡ Advanced Mode")
        self._adv_btn.clicked.connect(self._on_advanced_mode)
        left_layout.addWidget(self._adv_btn)

        left_layout.addStretch()
        splitter.addWidget(left)

        # ── RIGHT panel ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_layout.addWidget(QLabel("Result"))
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFont(QFont("Consolas", 12))
        self._result_text.setStyleSheet("""
            QTextEdit {
                background: #1a1a2e;
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self._result_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_layout.addWidget(self._result_text)

        # Action buttons
        actions = QHBoxLayout()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.clicked.connect(self._on_copy)
        self._copy_btn.setEnabled(False)
        actions.addWidget(self._copy_btn)

        self._save_btn = QPushButton("Save to File")
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setEnabled(False)
        actions.addWidget(self._save_btn)

        right_layout.addLayout(actions)
        splitter.addWidget(right)

        splitter.setSizes([380, 700])
        self._tabs.addTab(tab, "🔍 OCR")

    # ──────────────── Hardware tab ────────────────

    def _setup_hardware_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)

        # Title + refresh button
        header = QHBoxLayout()
        title = QLabel("System Hardware Info")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #c0c0ff;")
        header.addWidget(title)
        header.addStretch()
        refresh_btn = QPushButton("Refresh Detection")
        refresh_btn.clicked.connect(self._refresh_hardware)
        refresh_btn.setStyleSheet("background: #7c3aed; color: white; border-radius: 6px; padding: 6px 16px;")
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        # Info display
        self._hw_text = QTextEdit()
        self._hw_text.setReadOnly(True)
        self._hw_text.setFont(QFont("Consolas", 13))
        self._hw_text.setStyleSheet("""
            QTextEdit {
                background: #1a1a2e;
                color: #c0c0d0;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        self._hw_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self._hw_text)

        self._refresh_hardware()
        self._tabs.addTab(tab, "🖥 Hardware")

    def _refresh_hardware(self):
        try:
            hw = detect()
            lines = [
                "╔══════════════════════════════════════╗",
                "║        SYSTEM  HARDWARE  INFO        ║",
                "╚══════════════════════════════════════╝",
                "",
                "── GPU ──",
                f"  Device          : {hw.gpu_name or 'N/A'}",
                f"  Count           : {hw.gpu_count}",
                f"  VRAM Total      : {hw.vram_total_mb} MB  ({hw.vram_total_mb / 1024:.1f} GB)",
                f"  VRAM Free       : {hw.vram_free_mb} MB",
                f"  CUDA Version    : {hw.cuda_version or 'N/A'}",
                f"  FP16 Support    : {'Yes' if hw.supports_fp16 else 'No'}",
                "",
                "── CPU / RAM ──",
                f"  CPU Cores       : {hw.cpu_cores}",
                f"  RAM Total       : {hw.ram_total_mb} MB  ({hw.ram_total_mb / 1024:.1f} GB)",
                "",
                "── Recommended ──",
                f"  Model Tier      : {hw.recommended_tier.upper()}",
                "",
            ]
            # Recommend presets
            lines.append("── Recommended Presets ──")
            for et in self._provider.list_engine_types():
                try:
                    p = self._provider.recommend_preset(et, hw)
                    lines.append(f"  [{p.engine_type}] {p.name}  (needs {p.recommended_vram_mb}MB)")
                except ValueError:
                    lines.append(f"  [{et}] No suitable preset")

            self._hw_text.setHtml(
                "<pre style='font-family:Consolas; font-size:13px; line-height:1.6; color:#c0c0d0;'>"
                + "\n".join(lines)
                + "</pre>"
            )
        except Exception as e:
            self._hw_text.setPlainText(f"Hardware detection failed: {e}")

    # ──────────────── Downloads tab ────────────────

    def _setup_downloads_tab(self):
        self._dl_tab = QWidget()
        layout = QVBoxLayout(self._dl_tab)
        layout.setContentsMargins(16, 16, 16, 16)

        # Log output — terminal style (create first so header can reference it)
        self._dl_log = QTextEdit()
        self._dl_log.setReadOnly(True)
        self._dl_log.setFont(QFont("Consolas", 12))
        self._dl_log.setStyleSheet("""
            QTextEdit {
                background: #0a0a14;
                color: #00ff88;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        self._dl_log.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._dl_log.setPlaceholderText("Download log will appear here...\n\nStart a download from the Models tab.")

        # Header
        header = QHBoxLayout()
        title = QLabel("Download Log")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #c0c0ff;")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._dl_log.clear)
        clear_btn.setStyleSheet("background: #2a2a3e; color: #c0c0d0; border-radius: 6px; padding: 6px 16px;")
        header.addWidget(clear_btn)
        layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        layout.addWidget(self._dl_log)

        # Progress bar + label
        prog_layout = QHBoxLayout()
        self._dl_progress = QProgressBar()
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(0)
        self._dl_progress.setVisible(False)
        self._dl_progress.setMaximumHeight(16)
        self._dl_progress.setFormat("Waiting...")
        prog_layout.addWidget(self._dl_progress, 1)

        self._dl_progress_label = QLabel("")
        self._dl_progress_label.setStyleSheet("color: #888; font-size: 12px;")
        self._dl_progress_label.setVisible(False)
        prog_layout.addWidget(self._dl_progress_label)

        layout.addLayout(prog_layout)

        self._tabs.addTab(self._dl_tab, "📥 Downloads")

    # ──────────────── Models tab ────────────────

    def _setup_models_tab(self):
        self._models_tab = QWidget()
        layout = QVBoxLayout(self._models_tab)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QHBoxLayout()
        title = QLabel("Model Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #c0c0ff;")
        header.addWidget(title)

        # Engine filter
        header.addStretch()
        header.addWidget(QLabel("Filter:"))
        self._model_filter = QComboBox()
        self._model_filter.addItem("All", None)
        self._model_filter.addItem("General OCR", "general_ocr")
        self._model_filter.addItem("Document Parsing", "document_parsing")
        self._model_filter.currentIndexChanged.connect(self._refresh_models)
        header.addWidget(self._model_filter)

        self._model_refresh_btn = QPushButton("Refresh List")
        self._model_refresh_btn.clicked.connect(self._refresh_models)
        self._model_refresh_btn.setStyleSheet("background: #7c3aed; color: white; border-radius: 6px; padding: 6px 16px;")
        header.addWidget(self._model_refresh_btn)

        self._model_remote_btn = QPushButton("Fetch from Server")
        self._model_remote_btn.clicked.connect(self._on_refresh_remote)
        self._model_remote_btn.setStyleSheet("background: #059669; color: white; border-radius: 6px; padding: 6px 16px;")
        self._model_remote_btn.setToolTip("Force re-fetch model catalog from henryborner.top")
        header.addWidget(self._model_remote_btn)
        layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        # Tree widget
        self._model_tree = QTreeWidget()
        self._model_tree.setHeaderLabels(["", "Engine", "Role", "Model", "Tier", "Cached"])
        self._model_tree.setIndentation(10)
        self._model_tree.setColumnWidth(0, 80)
        self._model_tree.setColumnWidth(1, 120)
        self._model_tree.setColumnWidth(2, 160)
        self._model_tree.setColumnWidth(3, 280)
        self._model_tree.setColumnWidth(4, 70)
        self._model_tree.setColumnWidth(5, 110)
        self._model_tree.setAlternatingRowColors(True)
        self._model_tree.setRootIsDecorated(True)
        header_view = self._model_tree.header()
        header_view.setStretchLastSection(False)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._model_tree)

        # Bottom actions
        bottom = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All Uncached")
        self._select_all_btn.clicked.connect(self._select_all_uncached)
        bottom.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.clicked.connect(self._deselect_all)
        bottom.addWidget(self._deselect_all_btn)

        bottom.addStretch()

        self._model_download_btn = QPushButton("⬇ Download Checked")
        self._model_download_btn.setStyleSheet("""
            QPushButton {
                background: #7c3aed;
                color: white;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #6d28d9; }
            QPushButton:disabled { background: #444; color: #888; }
        """)
        self._model_download_btn.clicked.connect(self._on_download_checked)
        bottom.addWidget(self._model_download_btn)

        self._model_status = QLabel("")
        self._model_status.setStyleSheet("color: #888;")
        bottom.addWidget(self._model_status)

        layout.addLayout(bottom)

        self._refresh_models()
        self._tabs.addTab(self._models_tab, "📦 Models")

    def _refresh_models(self):
        self._model_tree.clear()
        self._model_items.clear()

        engine_filter = self._model_filter.currentData()
        models = self._provider.list_models(engine_filter)

        # Group by engine → role
        groups: dict[str, dict[str, list]] = {}
        for m in sorted(models, key=lambda x: (x.engine_type, x.role, x.tier, x.name)):
            groups.setdefault(m.engine_type, {}).setdefault(m.role, []).append(m)

        for engine_type, roles in groups.items():
            eng_item = QTreeWidgetItem(["", engine_type, "", "", "", ""])
            eng_item.setFlags(eng_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            eng_font = eng_item.font(0)
            eng_font.setBold(True)
            eng_item.setFont(0, eng_font)
            self._model_tree.addTopLevelItem(eng_item)

            for role, role_models in roles.items():
                role_item = QTreeWidgetItem(["", "", role, f"({len(role_models)} models)", "", ""])
                role_item.setFlags(role_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                eng_item.addChild(role_item)

                for m in role_models:
                    cached = self._provider.is_model_cached(m)
                    row = QTreeWidgetItem([
                        "",
                        "",
                        "",
                        m.name,
                        m.tier.upper(),
                        "✅ Cached" if cached else "⬜ Not cached",
                    ])
                    row.setData(0, Qt.ItemDataRole.UserRole, m.id)
                    # Checkbox on column 0
                    row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    if cached:
                        row.setCheckState(0, Qt.CheckState.Unchecked)
                        row.setFlags(row.flags() & ~Qt.ItemFlag.ItemIsEnabled)  # can't check cached
                        row.setForeground(5, Qt.GlobalColor.green)
                    else:
                        row.setCheckState(0, Qt.CheckState.Unchecked)
                        row.setForeground(5, Qt.GlobalColor.gray)
                    role_item.addChild(row)
                    self._model_items[m.id] = row

            eng_item.setExpanded(True)

        total = len(models)
        cached_count = sum(1 for m in models if self._provider.is_model_cached(m))
        src = self._provider.catalog_source
        src_label = {"remote": "☁ Server", "cache": "📦 Cache", "local": "💻 Local"}.get(src, src)
        self._model_status.setText(
            f"{total} models  |  {cached_count} cached  |  {total - cached_count} downloadable  |  Source: {src_label}"
        )

    def _select_all_uncached(self):
        for item in self._model_items.values():
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(0, Qt.CheckState.Checked)

    def _deselect_all(self):
        for item in self._model_items.values():
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(0, Qt.CheckState.Unchecked)

    def _get_checked_model_ids(self) -> list[str]:
        ids = []
        for mid, item in self._model_items.items():
            if item.checkState(0) == Qt.CheckState.Checked:
                ids.append(mid)
        return ids

    def _set_models_tab_locked(self, locked: bool):
        """Lock/unlock the Models tab during download."""
        idx = self._tabs.indexOf(self._models_tab)
        self._tabs.setTabEnabled(idx, not locked)
        self._model_filter.setEnabled(not locked)
        self._model_refresh_btn.setEnabled(not locked)
        self._model_remote_btn.setEnabled(not locked)
        self._model_download_btn.setEnabled(not locked)
        self._select_all_btn.setEnabled(not locked)
        self._deselect_all_btn.setEnabled(not locked)

    def _on_refresh_remote(self):
        """Force re-fetch model catalog from server."""
        self._model_refresh_btn.setEnabled(False)
        self._model_remote_btn.setEnabled(False)
        self._model_status.setText("Fetching from server...")
        # Use a QThread to avoid blocking UI
        class _FetchThread(QThread):
            done = pyqtSignal(str)
            def run(self):
                provider = ProviderRegistry.get_default()
                if provider:
                    src = provider.refresh_remote_catalog()
                    self.done.emit(src)

        self._fetch_thread = _FetchThread()
        def _on_done(src: str):
            self._model_refresh_btn.setEnabled(True)
            self._model_remote_btn.setEnabled(True)
            self._refresh_models()
            label = {"remote": "☁ Server", "cache": "📦 Cache", "local": "💻 Local"}.get(src, src)
            self._model_status.setText(self._model_status.text() + f" (updated via {label})")

        self._fetch_thread.done.connect(_on_done)
        self._fetch_thread.start()

    def _on_tab_changed(self, index: int):
        """Prevent switching to locked models tab."""
        pass  # setTabEnabled handles this natively

    def _on_download_checked(self):
        checked_ids = self._get_checked_model_ids()
        if not checked_ids:
            QMessageBox.information(self, "No Selection",
                "Please check at least one model to download.\nUse the checkboxes in the first column.")
            return

        all_models = self._provider.list_models()
        to_download = [m for m in all_models if m.id in checked_ids]

        # Lock models tab
        self._set_models_tab_locked(True)
        self._model_status.setText(f"Downloading {len(to_download)} model(s)...")

        # Clear downloads log and show progress
        self._dl_log.clear()
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(0)
        self._dl_progress.setFormat("Preparing...")
        self._dl_progress.setVisible(True)
        self._dl_progress_label.setText("")
        self._dl_progress_label.setVisible(True)

        # Switch to Downloads tab
        dl_idx = self._tabs.indexOf(self._dl_tab)
        self._tabs.setCurrentIndex(dl_idx)

        # Start worker
        self._download_worker = DownloadWorker(to_download)
        self._download_worker.log.connect(self._on_dl_log)
        self._download_worker.progress.connect(self._on_dl_progress)
        self._download_worker.model_done.connect(self._on_dl_model_done)
        self._download_worker.all_done.connect(self._on_dl_all_done)
        self._download_worker.start()

    def _on_dl_log(self, msg: str):
        self._dl_log.append(msg)
        # Auto-scroll to bottom
        scrollbar = self._dl_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_dl_progress(self, current: int, total: int):
        """Update progress bar from real HTTP download bytes."""
        if total > 0:
            pct = int(current * 100 / total)
            self._dl_progress.setRange(0, 100)
            self._dl_progress.setValue(pct)
            self._dl_progress.setFormat(f"{pct}%")
            self._dl_progress_label.setText(
                f"{current / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
            )
        else:
            mb = current / 1024 / 1024
            self._dl_progress.setRange(0, 0)  # indeterminate
            self._dl_progress.setFormat(f"{mb:.1f} MB")
            self._dl_progress_label.setText(f"{mb:.1f} MB downloaded")

    def _on_dl_model_done(self, name: str, success: bool):
        # Update tree item status if visible
        pass

    def _on_dl_all_done(self, ok_count: int, fail_count: int):
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(100)
        self._dl_progress.setFormat("Complete!")
        self._set_models_tab_locked(False)
        self._refresh_models()
        self._model_status.setText(f"Download complete: {ok_count} OK, {fail_count} failed")

    # ── advanced mode ──────────────────────────────────────────────

    def _on_advanced_mode(self):
        reply = QMessageBox.warning(
            self,
            "Advanced Mode",
            "You are entering Advanced Mode.\n\n"
            "• VRAM / hardware checks are DISABLED\n"
            "• All models from both OCR and Document Parsing are shown\n"
            "• You are responsible for knowing what your GPU can handle\n\n"
            "Running an 18GB model on a 4GB GPU may crash your system.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create advanced tab if not already open
        if not hasattr(self, "_adv_tab"):
            self._setup_advanced_tab()
        idx = self._tabs.indexOf(self._adv_tab)
        self._tabs.setCurrentIndex(idx)

    def _setup_advanced_tab(self):
        from PyQt6.QtWidgets import QScrollArea

        self._adv_tab = QWidget()
        self._adv_input_path: str | None = None
        root = QHBoxLayout(self._adv_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── LEFT (scrollable) ──
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)

        # Warning banner
        banner = QLabel("⚠ ADVANCED MODE — No VRAM checks. Pick any model per role.")
        banner.setStyleSheet("""
            QLabel {
                background: #3d2000;
                color: #f59e0b;
                border: 1px solid #f59e0b;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ll.addWidget(banner)

        # Drop zone
        drop_label = QLabel("Drag image or PDF here\nor click to browse")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setMinimumHeight(100)
        drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #f59e0b;
                border-radius: 12px;
                background: #1e1e2e;
                color: #888;
                font-size: 13px;
                padding: 18px;
            }
        """)
        drop_label.mousePressEvent = lambda e: self._adv_select_file()
        ll.addWidget(drop_label)
        self._adv_drop_label = drop_label

        # Scrollable model pickers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #0f0f1a; }")

        picker_widget = QWidget()
        picker_layout = QVBoxLayout(picker_widget)
        picker_layout.setContentsMargins(0, 0, 0, 0)

        # Group models by role
        all_models = self._provider.list_models()
        roles: dict[str, list] = {}
        for m in all_models:
            roles.setdefault(m.role, []).append(m)

        self._adv_role_combos: dict[str, QComboBox] = {}
        ROLE_LABELS = {
            "text_detection": "Text Detection",
            "text_recognition": "Text Recognition",
            "doc_text_orientation": "Doc Orientation",
            "textline_orientation": "Textline Orientation",
            "image_unwarping": "Image Unwarping",
            "layout_detection": "Layout Detection",
            "layout_analysis": "Layout Analysis",
            "formula_recognition": "Formula Recognition",
            "table_structure_recognition": "Table Structure",
            "table_cells_detection": "Table Cells Detection",
            "seal_text_detection": "Seal Text Detection",
            "chart_parsing": "Chart Parsing",
            "table_classification": "Table Classification",
        }

        for role in sorted(roles.keys()):
            role_models = sorted(roles[role], key=lambda x: (x.tier, x.name))
            gb = QGroupBox(ROLE_LABELS.get(role, role))
            gb.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #333;
                    border-radius: 4px;
                    margin-top: 6px;
                    padding-top: 12px;
                    font-size: 11px;
                    color: #888;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 6px; }
            """)
            gb_layout = QVBoxLayout(gb)
            combo = QComboBox()
            combo.addItem("(none)", None)
            for m in role_models:
                cached = self._provider.is_model_cached(m)
                prefix = "✅ " if cached else "⬜ "
                combo.addItem(f"{prefix}{m.name} [{m.tier}]", m.id)
            gb_layout.addWidget(combo)
            picker_layout.addWidget(gb)
            self._adv_role_combos[role] = combo

        picker_layout.addStretch()
        scroll.setWidget(picker_widget)
        ll.addWidget(scroll, 1)

        # Format selector
        ll.addWidget(QLabel("Output Format"))
        self._adv_format_combo = QComboBox()
        ll.addWidget(self._adv_format_combo)
        self._adv_populate_formats()

        # Run button
        self._adv_run_btn = QPushButton("Run OCR")
        self._adv_run_btn.setMinimumHeight(40)
        self._adv_run_btn.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: #1a1a2e;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover { background: #d97706; }
            QPushButton:disabled { background: #444; color: #888; }
        """)
        self._adv_run_btn.clicked.connect(self._adv_on_run)
        ll.addWidget(self._adv_run_btn)

        # Progress
        self._adv_progress = QProgressBar()
        self._adv_progress.setRange(0, 0)
        self._adv_progress.setVisible(False)
        ll.addWidget(self._adv_progress)
        splitter.addWidget(left)

        # ── RIGHT ──
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.addWidget(QLabel("Result"))
        self._adv_result_text = QTextEdit()
        self._adv_result_text.setReadOnly(True)
        self._adv_result_text.setFont(QFont("Consolas", 12))
        self._adv_result_text.setStyleSheet("""
            QTextEdit {
                background: #1a1a2e;
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self._adv_result_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        rl.addWidget(self._adv_result_text)
        splitter.addWidget(right)

        splitter.setSizes([400, 680])
        self._tabs.addTab(self._adv_tab, "⚡ Advanced")

    def _adv_populate_formats(self):
        self._adv_format_combo.clear()
        for f in FormatterRegistry.list_all():
            self._adv_format_combo.addItem(f.label, f.format_id)

    def _adv_check_run(self):
        # Always enable in advanced mode — user is responsible
        self._adv_run_btn.setEnabled(self._adv_input_path is not None)

    def _adv_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image or PDF", "",
            "Images & PDF (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.pdf);;All Files (*)"
        )
        if path:
            self._adv_input_path = path
            self._adv_drop_label.setText(f"Selected:\n{Path(path).name}")
            self._adv_drop_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #f59e0b;
                    border-radius: 12px;
                    background: #1e1e2e;
                    color: #f59e0b;
                    font-size: 14px;
                    padding: 24px;
                }
            """)
            self._adv_check_run()

    def _adv_on_run(self):
        if not self._adv_input_path:
            return

        # Build model selections from role combos
        model_selections = {}
        for role, combo in self._adv_role_combos.items():
            mid = combo.currentData()
            if mid:
                model_selections[role] = mid.split(".")[-1]  # "paddleocr.text_det.PP-OCRv6_medium_det" -> "PP-OCRv6_medium_det"

        if not model_selections:
            QMessageBox.warning(self, "No Models", "Please select at least one model.")
            return

        # Determine engine type from role
        _OCR_ROLES = {
            "text_detection", "text_recognition", "doc_text_orientation",
            "textline_orientation", "image_unwarping",
        }
        selected_roles = set(model_selections.keys())
        if selected_roles & _OCR_ROLES and selected_roles - _OCR_ROLES:
            # Mixed — use document_parsing since it handles both
            engine_type = "document_parsing"
        elif selected_roles & _OCR_ROLES:
            engine_type = "general_ocr"
        else:
            engine_type = "document_parsing"

        preset = EnginePreset(
            id="__advanced__",
            name="Advanced Custom",
            engine_type=engine_type,
            recommended_vram_mb=0,
            model_selections=model_selections,
        )

        fmt_id = self._adv_format_combo.currentData()

        self._adv_run_btn.setEnabled(False)
        self._adv_progress.setVisible(True)
        self._adv_result_text.clear()

        self._adv_worker = OCRWorker(engine_type, self._adv_input_path, "__advanced__", fmt_id, "gpu", preset=preset)
        self._adv_worker.finished.connect(self._adv_on_finished)
        self._adv_worker.error.connect(self._adv_on_error)
        self._adv_worker.start()

    def _adv_on_finished(self, output: str):
        self._adv_result_text.setPlainText(output)
        self._adv_progress.setVisible(False)
        self._adv_run_btn.setEnabled(True)

    def _adv_on_error(self, msg: str):
        self._adv_result_text.setPlainText(f"ERROR: {msg}")
        self._adv_progress.setVisible(False)
        self._adv_run_btn.setEnabled(True)

    def _run_remote(self):
        """Send image to remote server for OCR."""
        url = self._remote_url.text().strip()
        if not url:
            QMessageBox.warning(self, "No Server", "Please enter the remote server URL.")
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._result_text.clear()

        self._remote_worker = RemoteWorker(url, self._input_path)
        self._remote_worker.finished.connect(self._on_finished)
        self._remote_worker.error.connect(self._on_error)
        self._remote_worker.start()

    # ── drag & drop ───────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._set_input(path)

    def _on_drop_click(self, _event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image or PDF", "",
            "Images & PDF (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.pdf);;All Files (*)"
        )
        if path:
            self._set_input(path)

    def _set_input(self, path: str):
        self._input_path = path
        name = Path(path).name
        self._drop_zone.set_selected_text(name)
        self._run_btn.setEnabled(True)
        self._check_run_enabled()

    # ── presets ───────────────────────────────────────────────────

    def _on_engine_changed(self):
        self._populate_presets()
        self._check_run_enabled()

    def _populate_presets(self):
        engine_type = self._engine_combo.currentData()
        self._preset_combo.clear()
        self._format_combo.clear()

        hw = detect()
        presets = self._provider.list_presets(engine_type)
        recommended = self._provider.recommend_preset(engine_type, hw)

        # Build a list of VRAM-viable presets
        viable_ids = set()
        for p in presets:
            if p.recommended_vram_mb <= hw.vram_total_mb:
                viable_ids.add(p.id)

        for i, p in enumerate(presets):
            fits = p.id in viable_ids
            label = f"{p.name} ({p.recommended_vram_mb / 1000:.1f} GB)"
            if not fits:
                label += " [insufficient VRAM]"
            if p.id == recommended.id:
                label += " *"
            self._preset_combo.addItem(label, p.id)
            if not fits:
                # Gray out the item in dropdown (model index)
                model = self._preset_combo.model()
                item = model.item(i)
                item.setEnabled(False)

            # Select best viable
            if p.id == recommended.id and fits:
                self._preset_combo.setCurrentIndex(i)

        # Fallback: select first viable
        if self._preset_combo.currentData() not in viable_ids:
            for j in range(self._preset_combo.count()):
                if self._preset_combo.itemData(j) in viable_ids:
                    self._preset_combo.setCurrentIndex(j)
                    break

        formats = FormatterRegistry.list_for_engine(engine_type)
        for f in formats:
            self._format_combo.addItem(f.label, f.format_id)

        self._viable_presets = viable_ids
        self._check_run_enabled()

    def _on_remote_toggled(self, checked: bool):
        self._remote_url.setVisible(checked)

    # ── run ───────────────────────────────────────────────────────

    def _check_run_enabled(self):
        """Disable Run button if selected preset doesn't fit VRAM."""
        current = self._preset_combo.currentData()
        if current and current not in self._viable_presets:
            if self._remote_cb.isChecked():
                self._run_btn.setEnabled(self._input_path is not None)
                self._run_btn.setText("Run via Remote")
            else:
                self._run_btn.setEnabled(False)
                self._run_btn.setText("Insufficient VRAM")
        else:
            self._run_btn.setEnabled(self._input_path is not None)
            self._run_btn.setText("Run OCR")

    def _on_run(self):
        if not self._input_path:
            QMessageBox.warning(self, "No Input", "Please select or drop an image first.")
            return

        current = self._preset_combo.currentData()
        if current and current not in self._viable_presets:
            if self._remote_cb.isChecked():
                self._run_remote()
                return
            QMessageBox.critical(self, "VRAM Insufficient",
                "This preset requires more GPU memory than your device has.\n"
                "Enable ☁ Remote to send to cloud server.")
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._result_text.clear()

        engine_type = self._engine_combo.currentData()
        preset_id = self._preset_combo.currentData()
        format_id = self._format_combo.currentData()

        self._worker = OCRWorker(engine_type, self._input_path, preset_id, format_id, "gpu")
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, output: str):
        self._result_text.setPlainText(output)
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._copy_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._result_text.setPlainText(f"ERROR: {msg}")
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    # ── copy / save ───────────────────────────────────────────────

    def _on_copy(self):
        text = self._result_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Copied", "Result copied to clipboard.")

    def _on_save(self):
        text = self._result_text.toPlainText()
        if not text:
            return
        fmt_id = self._format_combo.currentData()
        fmt = FormatterRegistry.get(fmt_id)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", f"result{fmt.file_extension}",
            f"*{fmt.file_extension}"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Saved", f"Saved to {path}")
