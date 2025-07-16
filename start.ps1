# AI Digit Recognizer - PowerShell Launcher
# Простой запуск программы распознавания цифр

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    🔢 AI Digit Recognizer - Launcher" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Проверяем наличие виртуальной среды
if (Test-Path "..\.venv_312\Scripts\python.exe") {
    Write-Host "✅ Найдена виртуальная среда .venv_312" -ForegroundColor Green
    Write-Host "🚀 Запуск программы..." -ForegroundColor Yellow
    Write-Host ""
    & "..\.venv_312\Scripts\python.exe" run.py
} elseif (Test-Path "..\.venv\Scripts\python.exe") {
    Write-Host "✅ Найдена виртуальная среда .venv" -ForegroundColor Green
    Write-Host "🚀 Запуск программы..." -ForegroundColor Yellow
    Write-Host ""
    & "..\.venv\Scripts\python.exe" run.py
} else {
    Write-Host "❌ Виртуальная среда не найдена!" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Попробуйте:" -ForegroundColor Yellow
    Write-Host "1. python run.py"
    Write-Host "2. python3 run.py"
    Write-Host "3. py run.py"
    Write-Host ""
    Write-Host "Если не работает, установите Python и зависимости:" -ForegroundColor Yellow
    Write-Host "pip install -r requirements.txt"
    Write-Host ""

    # Пробуем запустить обычным python
    try {
        python run.py
    } catch {
        Write-Host "❌ Python не найден в системе!" -ForegroundColor Red
        Write-Host "Установите Python 3.8+ с официального сайта python.org" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Программа завершена. Нажмите Enter для выхода..." -ForegroundColor Gray
Read-Host
