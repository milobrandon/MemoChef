@echo off
title Memo Chef
echo.
echo  ============================
echo   Memo Chef - Local Launcher
echo  ============================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Create venv if missing
if not exist ".venv" (
    echo  Setting up environment (first run only)...
    python -m venv .venv
)

:: Activate and install deps
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt >nul 2>&1

:: Set up secrets on first run
if not exist ".streamlit\secrets.toml" (
    if exist ".streamlit\secrets.toml.local" (
        copy ".streamlit\secrets.toml.local" ".streamlit\secrets.toml" >nul
        echo.
        echo  [SETUP] Created .streamlit\secrets.toml from template.
        echo  Open it and paste your Anthropic API key, then relaunch.
        echo  Get a key at https://console.anthropic.com/settings/keys
        echo.
        start notepad ".streamlit\secrets.toml"
        pause
        exit /b 0
    ) else (
        echo.
        echo  [ERROR] Missing .streamlit\secrets.toml and no template found.
        echo  Copy secrets.toml.example and fill in your values.
        echo.
        pause
        exit /b 1
    )
)

:: Verify API key was set
findstr /C:"PASTE_YOUR_KEY_HERE" ".streamlit\secrets.toml" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  [SETUP] Please paste your Anthropic API key in .streamlit\secrets.toml
    start notepad ".streamlit\secrets.toml"
    pause
    exit /b 0
)

echo  Starting Memo Chef...
echo  Open http://localhost:8501 in your browser
echo  Press Ctrl+C to stop
echo.
streamlit run app.py --server.port 8501
