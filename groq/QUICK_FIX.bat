@echo off
chcp 65001 >nul
echo ============================================================
echo iPsychiatrist - Quick Fix Script
echo ============================================================
echo.

echo [1/4] Stopping Streamlit...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/4] Removing old vectors...
if exist vectors rmdir /s /q vectors
echo [OK] Old vectors removed

echo.
echo [3/4] Backing up app.py...
if exist app.py copy /y app.py app_backup.py >nul
if exist app_fixed.py copy /y app_fixed.py app.py >nul
echo [OK] App updated

echo.
echo [4/4] Installing packages (this may take a few minutes)...
py -3.12 -m pip install --quiet --upgrade langchain langchain-core langchain-community langchain-openai langchain-text-splitters langchain-groq langchain-classic langchain-huggingface sentence-transformers torch transformers "numpy<2" "pillow<11" "packaging<25" scikit-learn faiss-cpu pypdf python-dotenv streamlit groq

if %ERRORLEVEL% EQU 0 (
    echo [OK] All packages installed
    echo.
    echo ============================================================
    echo Setup Complete!
    echo ============================================================
    echo.
    echo Next: Run "streamlit run app.py"
    echo.
    echo Note: First run will download the AI model (~400MB)
    echo       and create embeddings (5-10 minutes)
    echo.
) else (
    echo [ERROR] Package installation failed
    echo        Close all Streamlit windows and try again
)

pause
