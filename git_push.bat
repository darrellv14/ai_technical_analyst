@echo off
echo ========================================
echo Git Push to GitHub
echo ========================================
echo.

echo [1/4] Checking git status...
git status
echo.

echo [2/4] Adding all files...
git add .
echo.

echo [3/4] Committing changes...
set /p commit_msg="Enter commit message: "
git commit -m "%commit_msg%"
echo.

echo [4/4] Pushing to GitHub...
git push origin main
echo.

if errorlevel 1 (
    echo.
    echo ERROR: Push failed!
    echo Possible solutions:
    echo 1. Check your internet connection
    echo 2. Make sure you have push access to the repository
    echo 3. Run: git remote -v to check remote URL
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Successfully pushed to GitHub!
echo ========================================
echo.
echo Next step: Deploy to Streamlit Cloud
echo Visit: https://share.streamlit.io/
echo.
pause
