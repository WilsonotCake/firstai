#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI Digit Recognizer - Launcher
Простой запуск программы распознавания цифр
"""

import sys
import os

def main():
    """Запуск программы распознавания цифр"""
    print("🔢 AI Digit Recognizer")
    print("=" * 30)

    try:
        # Импортируем основную программу
        from digit_recognizer import main as recognizer_main

        print("✅ Запуск системы распознавания...")
        recognizer_main()

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("\n💡 Убедитесь что:")
        print("1. Установлены все зависимости: pip install -r requirements.txt")
        print("2. Файл digit_recognizer.py находится в той же папке")

    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        print("\n💡 Попробуйте:")
        print("1. Перезапустить программу")
        print("2. Проверить наличие папки models/")

if __name__ == "__main__":
    main()
