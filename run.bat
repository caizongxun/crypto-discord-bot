@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Crypto Discord Bot - Windows Launcher
echo ============================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% found
echo.

REM Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Install/upgrade requirements
echo Checking dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo ✓ Dependencies installed
echo.

REM Check .env file
if not exist ".env" (
    echo Error: .env file not found
    echo Creating .env from .env.example...
    copy .env.example .env
    echo ✓ .env created (edit with your Discord token!)
    echo.
    echo Current .env contents:
    type .env
    echo.
    echo Please edit .env and run this script again
    pause
    exit /b 1
)

REM Check Discord token
for /f "tokens=2 delims==" %%i in ('findstr "DISCORD_TOKEN=" .env') do set DISCORD_TOKEN=%%i
if "%DISCORD_TOKEN%"==" " (
    echo Error: DISCORD_TOKEN not set in .env
    pause
    exit /b 1
)
echo ✓ Configuration valid
echo.

echo ============================================
echo Starting services...
echo ============================================
echo.

echo Starting Discord bot...
start "Crypto Discord Bot" python bot.py
echo ✓ Discord bot started
echo.

REM Check for --with-dashboard or -d flag
if "%1"=="--with-dashboard" goto start_dashboard
if "%1"=="-d" goto start_dashboard
goto skip_dashboard

:start_dashboard
echo Starting web dashboard...
for /f "tokens=2 delims==" %%i in ('findstr "DASHBOARD_PORT=" .env') do set DASHBOARD_PORT=%%i
if "!DASHBOARD_PORT!"==" " set DASHBOARD_PORT=5000
start "Crypto Dashboard" python dashboard.py
echo ✓ Dashboard started on http://localhost:!DASHBOARD_PORT!
echo.
goto done

:skip_dashboard
echo To start dashboard, run: run.bat --with-dashboard
echo.

:done
echo ============================================
echo ✓ All services started!
echo ============================================
echo.
echo Bot window opened above.
echo You can close this window when done.
echo Press Ctrl+C in the bot window to stop.
echo.
pause
