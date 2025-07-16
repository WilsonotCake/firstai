#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Installation Test Script
Tests if all dependencies are installed and working correctly
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - FAILED: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow specifically"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - OK")

        # Test if GPU is available (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU available: {len(gpus)} device(s)")
        else:
            print("   Running on CPU (this is fine)")

        return True
    except Exception as e:
        print(f"❌ TensorFlow - FAILED: {e}")
        return False

def test_basic_functionality():
    """Test basic AI functionality"""
    try:
        import numpy as np

        # Test numpy array creation
        test_array = np.array([1, 2, 3])

        # Test basic math
        result = np.sum(test_array)

        if result == 6:
            print("✅ Basic NumPy operations - OK")
            return True
        else:
            print("❌ Basic NumPy operations - FAILED")
            return False

    except Exception as e:
        print(f"❌ Basic functionality test - FAILED: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    try:
        import cv2
        from PIL import Image
        import numpy as np

        # Create a simple test image
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[:, :] = [255, 255, 255]  # White image

        # Test PIL
        pil_image = Image.fromarray(test_image)

        # Test OpenCV basic operation
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

        print("✅ Image processing (OpenCV + PIL) - OK")
        return True

    except Exception as e:
        print(f"❌ Image processing - FAILED: {e}")
        return False

def test_tkinter():
    """Test Tkinter GUI capabilities"""
    try:
        import tkinter as tk

        # Create a test window (don't show it)
        root = tk.Tk()
        root.withdraw()  # Hide the window

        # Test basic widgets
        label = tk.Label(root, text="Test")
        button = tk.Button(root, text="Test Button")

        root.destroy()  # Clean up

        print("✅ Tkinter GUI - OK")
        return True

    except Exception as e:
        print(f"❌ Tkinter GUI - FAILED: {e}")
        return False

def test_project_files():
    """Test if project files exist and can be imported"""
    import os

    files_to_check = [
        'utils.py',
        'digit_recognition.py',
        'example_usage.py',
        'requirements.txt',
        'README.md'
    ]

    print("\n📁 Checking project files:")
    all_files_exist = True

    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"✅ {filename} - Found")
        else:
            print(f"❌ {filename} - Missing")
            all_files_exist = False

    # Test imports from project files
    try:
        print("\n🔗 Testing project imports:")

        # Test utils.py import
        sys.path.insert(0, '.')
        import utils
        print("✅ utils.py - Can import")

        # Test if AI recognizer is available
        if hasattr(utils, 'ai_recognizer'):
            print("✅ AI recognizer - Available")
        else:
            print("❌ AI recognizer - Not found")

    except Exception as e:
        print(f"❌ Project imports - FAILED: {e}")
        all_files_exist = False

    return all_files_exist

def run_quick_ai_test():
    """Run a quick AI functionality test"""
    try:
        print("\n🤖 Quick AI Test:")

        import tensorflow as tf
        import numpy as np

        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Test prediction with dummy data
        dummy_input = np.random.random((1, 5))
        prediction = model.predict(dummy_input, verbose=0)

        print("✅ AI model creation and prediction - OK")
        return True

    except Exception as e:
        print(f"❌ AI functionality test - FAILED: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 AI Digit Recognition - Installation Test")
    print("=" * 50)

    # Test Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ is recommended")

    print("\n📦 Testing Dependencies:")

    # Test core dependencies
    tests = [
        ("numpy", "NumPy"),
        ("tensorflow", None),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]

    passed_tests = 0
    total_tests = len(tests) + 5  # +5 for additional tests

    # Test basic imports
    for module, display_name in tests:
        if module == "tensorflow":
            if test_tensorflow():
                passed_tests += 1
        else:
            if test_import(module, display_name):
                passed_tests += 1

    # Test Tkinter separately
    if test_tkinter():
        passed_tests += 1

    # Test basic functionality
    if test_basic_functionality():
        passed_tests += 1

    # Test image processing
    if test_image_processing():
        passed_tests += 1

    # Test project files
    if test_project_files():
        passed_tests += 1

    # Quick AI test
    if run_quick_ai_test():
        passed_tests += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Run 'python digit_recognition.py' for the GUI")
        print("2. Run 'python example_usage.py' for examples")
        print("3. Import utils in your own scripts")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTo fix issues:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ recommended)")
        print("3. Make sure all project files are present")

    # Additional recommendations
    print("\n💡 Performance Tips:")
    print("- First run will download and train the AI model (may take 2-5 minutes)")
    print("- Subsequent runs will be much faster")
    print("- Use clear, single-digit images for best results")

if __name__ == "__main__":
    main()
