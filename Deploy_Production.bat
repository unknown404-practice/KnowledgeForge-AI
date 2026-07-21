@echo off
color 0A
title KnowledgeForge AI - Production Deploy

echo =======================================================
echo          KnowledgeForge AI - Production Deployment
echo =======================================================
echo.

:: Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not running!
    echo.
    echo To deploy KnowledgeForge AI in production, you need Docker Desktop.
    echo Please download and install it from: https://www.docker.com/products/docker-desktop/
    echo.
    echo Once installed and running, double-click this script again.
    echo.
    pause
    exit /b 1
)

echo [OK] Docker is installed.
echo.
echo Initializing Production Build...
echo.

:: Run Docker Compose
docker-compose up -d --build

if %ERRORLEVEL% equ 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] KnowledgeForge AI is now running in production!
    echo.
    echo You can access the Dashboard at: http://localhost:8501
    echo =======================================================
) else (
    echo.
    echo [ERROR] Docker Compose failed to build or start the container.
)

pause
