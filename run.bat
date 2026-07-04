@echo off
REM ── SIF Engine launcher ─────────────────────────────────────────────
REM Starts the web UI with the project venv (real models when available)
REM and opens it in your browser. Double-click, or run from a terminal.

cd /d "%~dp0"

REM Prefer the project venv (has the real models); fall back to system Python.
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting SIF Engine...
echo   Python: %PY%
echo   URL:    http://127.0.0.1:8000
echo.

REM Nudge the Ollama daemon so scene captions use the real VLM (harmless if up).
start "" /b cmd /c "ollama list >nul 2>&1"

REM Open the browser ~6s after the server starts (ping is a reliable delay;
REM the URL is unquoted so START opens it instead of treating it as a title).
start "" /b cmd /c "ping -n 7 127.0.0.1 >nul & start http://127.0.0.1:8000"

REM Run the server (blocks here; close this window or press Ctrl+C to stop).
"%PY%" -m sif.cli serve

echo.
echo SIF Engine stopped.
pause
