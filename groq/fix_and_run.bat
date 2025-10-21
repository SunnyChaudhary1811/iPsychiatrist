@echo off
echo ========================================
echo iPsychiatrist - Fix and Run Script
echo ========================================
echo.

echo Step 1: Stopping any running Streamlit instances...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Running setup script...
py -3.12 setup_fix.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Setup completed successfully!
    echo ========================================
    echo.
    echo Starting Streamlit app...
    echo.
    py -3.12 -m streamlit run app.py
) else (
    echo.
    echo ========================================
    echo Setup failed. Please check errors above.
    echo ========================================
    pause
)
