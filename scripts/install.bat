@echo off
chcp 65001 >nul
title Scribe OCR Toolkit — Installer

echo ============================================
echo   Scribe OCR Toolkit — One-Click Installer
echo ============================================
echo.

REM ── Step 1: Check Python ──
set PYTHON=
for /f "tokens=*" %%p in ('where python 2^>nul') do set PYTHON=%%p
if "%PYTHON%"=="" (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python 3.10 or later:
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Download and run the installer
    echo   3. CHECK "Add Python to PATH"
    echo   4. Run this script again
    echo.
    pause
    exit /b 1
)

echo [OK] Python found: %PYTHON%
python --version
echo.

REM ── Step 2: Create virtual environment ──
set VENV_DIR=%~dp0..\venv
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [*] Virtual environment already exists at %VENV_DIR%
) else (
    echo [*] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

REM ── Step 3: Upgrade pip ──
echo [*] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo.

REM ── Step 4: Install Scribe Frame (local) ──
echo [*] Installing scribe-frame...
"%VENV_DIR%\Scripts\pip.exe" install -e "%~dp0..\..\scribe_frame"
if errorlevel 1 (
    echo [ERROR] scribe-frame installation failed.
    pause
    exit /b 1
)
echo [OK] scribe-frame installed.
echo.

REM ── Step 5: Install Scribe ──
echo [*] Installing Scribe OCR Toolkit (with GUI)...
cd /d "%~dp0.."
"%VENV_DIR%\Scripts\pip.exe" install -e ".[gui]"
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Installation failed.
    echo Try running: "%VENV_DIR%\Scripts\pip.exe" install -e "%~dp0.."
    pause
    exit /b 1
)
echo [OK] Scribe installed successfully.
echo.

REM ── Step 6: Verify ──
echo [*] Verifying installation...
"%VENV_DIR%\Scripts\scribe.exe" --help >nul 2>&1
if errorlevel 1 (
    echo [WARNING] CLI verification failed, but GUI may still work.
) else (
    echo [OK] CLI works.
)
echo.

REM ── Step 7: Launch ──
echo ============================================
echo   Installation complete!
echo ============================================
echo.
echo Starting Scribe GUI...
echo.
"%VENV_DIR%\Scripts\scribe-gui.exe"

if errorlevel 1 (
    echo.
    echo [ERROR] GUI failed to start.
    echo You can try running it manually:
    echo   %VENV_DIR%\Scripts\scribe-gui.exe
    pause
)
