"""Scribe GUI launcher."""

import os
import sys

# Fix nvidia DLL path
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

from PyQt6.QtWidgets import QApplication
from scribe.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Scribe")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
