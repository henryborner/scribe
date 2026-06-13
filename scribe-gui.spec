# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['E:\\OCR\\scribe\\src\\scribe\\ui\\launch.py'],
    pathex=[],
    binaries=[],
    datas=[('E:\\OCR\\venv\\Lib\\site-packages\\paddleocr', 'paddleocr'), ('E:\\OCR\\venv\\Lib\\site-packages\\paddlex', 'paddlex')],
    hiddenimports=['paddle', 'paddleocr', 'paddlex', 'PyQt6'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='scribe-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='scribe-gui',
)
