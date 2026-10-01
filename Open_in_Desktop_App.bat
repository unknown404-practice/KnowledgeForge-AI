@echo off
echo =========================================================================
echo Opening KnowledgeForge AI inside Desktop Application (No LocalHost)...
echo =========================================================================
cd /d "%~dp0"

set "PYTHON_EXE=python"
echo Starting Ollama Backend...
start /B ollama serve >nul 2>&1
if exist "%APPDATA%\jupyterlab-desktop\jlab_server\python.exe" (
    set "PYTHON_EXE=%APPDATA%\jupyterlab-desktop\jlab_server\python.exe"
)

echo Using Python environment: %PYTHON_EXE%
"%PYTHON_EXE%" -c "import webview" 2>nul || (echo Installing pywebview... && "%PYTHON_EXE%" -m pip install pywebview)

"%PYTHON_EXE%" app\desktop.py
pause
