@echo off
setlocal
echo =========================================================================
echo Opening KnowledgeForge AI Native Desktop Edition...
echo =========================================================================
cd /d "%~dp0"

echo Starting Ollama Backend Engine...
start "" /B ollama serve >nul 2>&1

set "PYTHON_EXE=python"
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -c "import webview" 2>nul || (echo Installing pywebview... && "%PYTHON_EXE%" -m pip install pywebview)

echo Launching Desktop UI...
"%PYTHON_EXE%" app\desktop.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    pause
)
