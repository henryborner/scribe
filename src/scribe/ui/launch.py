"""Scribe GUI launcher."""

import os
import sys

# Fix nvidia DLL path
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_bin = os.path.join(_venv_root, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
if os.path.isdir(_nvidia_bin):
    try:
        os.add_dll_directory(_nvidia_bin)
    except AttributeError:
        pass
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

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
