@echo off
echo ================================================================================
echo GMAO AI Dashboard Launcher
echo ================================================================================
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open in your default browser.
echo Press Ctrl+C to stop the dashboard.
echo.
cd /d "%~dp0"
streamlit run streamlit_app.py
pause
