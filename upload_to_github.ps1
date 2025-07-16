# AI Digit Recognizer - GitHub Upload Script (PowerShell)
# Автоматическая загрузка проекта на GitHub

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    🚀 GitHub Upload Script" -ForegroundColor Yellow
Write-Host "    AI Digit Recognizer v3.0" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 Checking Git status..." -ForegroundColor Blue
git status
Write-Host ""

Write-Host "⚠️  IMPORTANT: Before running this script:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub.com"
Write-Host "2. Copy the repository URL"
Write-Host "3. Make sure you're logged into Git"
Write-Host ""

$repoUrl = Read-Host "📎 Enter your GitHub repository URL (e.g., https://github.com/username/ai-digit-recognizer.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "❌ Repository URL is required!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🔗 Adding remote repository..." -ForegroundColor Green
git remote add origin $repoUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠️  Remote already exists. Updating..." -ForegroundColor Yellow
    git remote set-url origin $repoUrl
}

Write-Host ""
Write-Host "🌿 Renaming branch to main..." -ForegroundColor Green
git branch -M main

Write-Host ""
Write-Host "📤 Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Upload failed! Common issues:" -ForegroundColor Red
    Write-Host "1. Check your internet connection"
    Write-Host "2. Verify repository URL is correct"
    Write-Host "3. Make sure you have push permissions"
    Write-Host "4. Try: git push --force-with-lease origin main"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "🎉 SUCCESS! Project uploaded to GitHub!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "🌟 Your repository is now available at:" -ForegroundColor Yellow
Write-Host "   $repoUrl" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 Next steps:" -ForegroundColor Blue
Write-Host "1. Add repository description and topics"
Write-Host "2. Upload screenshots to README"
Write-Host "3. Star your own repository ⭐"
Write-Host "4. Share with friends and colleagues"
Write-Host ""

Write-Host "🔄 To update in future:" -ForegroundColor Blue
Write-Host "   git add ."
Write-Host "   git commit -m `"Update description`""
Write-Host "   git push origin main"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Press Enter to open repository in browser..." -ForegroundColor Gray
Read-Host

# Try to open the repository in default browser
$browserUrl = $repoUrl -replace '\.git$', ''
try {
    Start-Process $browserUrl
    Write-Host "🌐 Opening repository in browser..." -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not open browser automatically" -ForegroundColor Yellow
    Write-Host "Please visit: $browserUrl" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🚀 Happy coding!" -ForegroundColor Green
