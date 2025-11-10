@echo off
echo ========================================
echo Brutal Stock Tool v10 - Installer
echo ========================================
echo.

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python tidak ditemukan. Install Python terlebih dahulu.
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
echo Ini mungkin memakan waktu beberapa menit...
echo.
python -m pip install --user streamlit yfinance pandas-ta lightgbm optuna plotly google-generativeai numpy pandas scikit-learn python-dotenv
if errorlevel 1 (
    echo.
    echo WARNING: Ada error saat install. Mencoba cara alternatif...
    python -m pip install --user --upgrade streamlit
)
echo.

echo [3/3] Installation complete!
echo.
echo ========================================
echo Untuk menjalankan aplikasi, klik:
echo   run_app.bat
echo ========================================
pause
