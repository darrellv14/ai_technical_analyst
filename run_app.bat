@echo off
echo ========================================
echo Brutal Stock Tool v10
echo Starting Streamlit App...
echo ========================================
echo.
echo Aplikasi akan terbuka di browser Anda.
echo Tekan Ctrl+C untuk stop aplikasi.
echo.

cd /d "%~dp0"
python -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo ERROR: Gagal menjalankan aplikasi.
    echo Pastikan sudah menjalankan install.bat terlebih dahulu.
    echo.
    pause
)
