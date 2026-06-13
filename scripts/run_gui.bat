@echo off
chcp 65001 >nul
title Scribe GUI Launcher

echo ============================================
echo   Scribe OCR Toolkit — GUI Launcher
echo ============================================
echo.

REM Check if venv exists
if not exist "%~dp0..\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo [*] Activating virtual environment...
call "%~dp0..\venv\Scripts\activate.bat"

echo [*] Launching Scribe GUI...
scribe-gui

if errorlevel 1 (
    echo.
    echo [ERROR] GUI failed to start.
    pause
)
