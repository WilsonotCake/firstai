@echo off
title Upload AI Digit Recognizer to GitHub
echo.
echo ========================================
echo    🚀 GitHub Upload Script
echo    AI Digit Recognizer v3.0
echo ========================================
echo.

echo 📋 Checking Git status...
git status
echo.

echo ⚠️  IMPORTANT: Before running this script:
echo 1. Create a new repository on GitHub.com
echo 2. Copy the repository URL
echo 3. Make sure you're logged into Git
echo.

set /p REPO_URL="📎 Enter your GitHub repository URL (e.g., https://github.com/username/ai-digit-recognizer.git): "

if "%REPO_URL%"=="" (
    echo ❌ Repository URL is required!
    pause
    exit /b 1
)

echo.
echo 🔗 Adding remote repository...
git remote add origin %REPO_URL%

if errorlevel 1 (
    echo.
    echo ⚠️  Remote already exists. Updating...
    git remote set-url origin %REPO_URL%
)

echo.
echo 🌿 Renaming branch to main...
git branch -M main

echo.
echo 📤 Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Upload failed! Common issues:
    echo 1. Check your internet connection
    echo 2. Verify repository URL is correct
    echo 3. Make sure you have push permissions
    echo 4. Try: git push --force-with-lease origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🎉 SUCCESS! Project uploaded to GitHub!
echo ========================================
echo.
echo 🌟 Your repository is now available at:
echo    %REPO_URL%
echo.
echo 📋 Next steps:
echo 1. Add repository description and topics
echo 2. Upload screenshots to README
echo 3. Star your own repository ⭐
echo 4. Share with friends and colleagues
echo.
echo 🔄 To update in future:
echo    git add .
echo    git commit -m "Update description"
echo    git push origin main
echo.
echo ========================================
echo Press any key to open repository in browser...
pause >nul

REM Try to open the repository in default browser
set "BROWSER_URL=%REPO_URL:~0,-4%"
start "" "%BROWSER_URL%"

echo.
echo 🚀 Happy coding!
