@echo off
title AI Digit Recognizer
echo.
echo ========================================
echo    🔢 AI Digit Recognizer - Launcher
echo ========================================
echo.

REM Проверяем наличие виртуальной среды
if exist "..\.venv_312\Scripts\python.exe" (
    echo ✅ Найдена виртуальная среда .venv_312
    echo 🚀 Запуск программы...
    echo.
    "..\.venv_312\Scripts\python.exe" run.py
) else if exist "..\.venv\Scripts\python.exe" (
    echo ✅ Найдена виртуальная среда .venv
    echo 🚀 Запуск программы...
    echo.
    "..\.venv\Scripts\python.exe" run.py
) else (
    echo ❌ Виртуальная среда не найдена!
    echo.
    echo 💡 Попробуйте:
    echo 1. python run.py
    echo 2. python3 run.py
    echo 3. py run.py
    echo.
    echo Если не работает, установите Python и зависимости:
    echo pip install -r requirements.txt
    echo.
    python run.py
)

echo.
echo ========================================
echo Программа завершена. Нажмите любую клавишу для выхода...
pause >nul
