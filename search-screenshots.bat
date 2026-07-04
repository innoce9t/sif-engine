@echo off
REM ── SIF Screenshot Search ───────────────────────────────────────────
REM Opens the web UI pointed at your indexed screenshots (screenshots_index).
REM Double-click the desktop shortcut, or run this file directly.

cd /d "%~dp0"

REM Prefer the project venv (has the real models); fall back to system Python.
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "DATA=%~dp0screenshots_index"
set "SIF_DATA=%DATA%"

echo Starting SIF Screenshot Search...
echo   Index: %DATA%
echo   URL:   http://127.0.0.1:8000
echo.

REM Nudge the Ollama daemon so captions/search use the real models (harmless if up).
start "" /b cmd /c "ollama list >nul 2>&1"

REM Open the browser a few seconds after the server starts.
start "" /b cmd /c "ping -n 7 127.0.0.1 >nul & start http://127.0.0.1:8000"

REM Run the server (blocks here; close this window or press Ctrl+C to stop).
"%PY%" -m sif.cli --data "%DATA%" serve

echo.
echo SIF Screenshot Search stopped.
pause
