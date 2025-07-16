#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage Script for AI Digit Recognition System
This script demonstrates how to use the digit recognition AI
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Import our AI modules
try:
    from utils import ai_recognize_digit, ai_recognizer, check_finals_status
    from digit_recognition import DigitRecognitionAI
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

def create_sample_digit_image(digit, filename):
    """Create a sample image with a digit for testing"""
    # Create a 100x100 white image
    img = Image.new('RGB', (100, 100), 'white')
    draw = ImageDraw.Draw(img)

    # Try to use a larger font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()

    # Draw the digit in black
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (100 - text_width) // 2
    y = (100 - text_height) // 2

    draw.text((x, y), text, fill='black', font=font)

    # Save the image
    img.save(filename)
    print(f"Created sample image: {filename}")
    return filename

def example_1_basic_recognition():
    """Example 1: Basic digit recognition from created images"""
    print("\n=== Example 1: Basic Digit Recognition ===")

    # Create sample images for digits 0-9
    test_images = []
    for digit in range(10):
        filename = f"sample_digit_{digit}.png"
        create_sample_digit_image(digit, filename)
        test_images.append(filename)

    # Test recognition on each image
    for image_path in test_images:
        print(f"\nTesting: {image_path}")
        digit, confidence = ai_recognize_digit(image_path)

        if digit is not None:
            print(f"✓ Recognized: {digit} (confidence: {confidence:.1f}%)")

            # Special case for our magic number 123
            if digit in [1, 2, 3]:
                print(f"  → This digit is part of the magic number 123!")
                if digit == 1:
                    print("  → Found digit 1 - first part of 123")
                elif digit == 2:
                    print("  → Found digit 2 - second part of 123")
                elif digit == 3:
                    print("  → Found digit 3 - third part of 123")
        else:
            print(f"✗ Recognition failed (confidence: {confidence:.1f}%)")

    # Cleanup
    for image_path in test_images:
        try:
            os.remove(image_path)
        except:
            pass

def example_2_gui_application():
    """Example 2: Launch the full GUI application"""
    print("\n=== Example 2: GUI Application ===")
    print("Launching the full AI Digit Recognition GUI...")
    print("Note: This will open a new window with the complete interface")

    try:
        app = DigitRecognitionAI()
        print("GUI application started. Close the window to continue...")
        app.run()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("You may need to install additional dependencies:")
        print("pip install tensorflow opencv-python pillow numpy")

def example_3_integration_with_original():
    """Example 3: Integration with original code logic"""
    print("\n=== Example 3: Integration with Original Logic ===")

    # Create an image with digit "123" written on it
    img = Image.new('RGB', (200, 80), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()

    draw.text((20, 20), "123", fill='black', font=font)
    filename = "number_123.png"
    img.save(filename)

    print(f"Created test image with '123': {filename}")

    # Try to recognize it (this might not work perfectly since it's multiple digits)
    digit, confidence = ai_recognize_digit(filename)

    print(f"AI Recognition result: digit={digit}, confidence={confidence:.1f}%")

    # Show original logic
    print("\nOriginal logic from utils.py:")
    result = check_finals_status()

    print(f"\nCombined result: AI saw '{digit}', original logic says finals is '{result}'")

    # Cleanup
    try:
        os.remove(filename)
    except:
        pass

def example_4_advanced_preprocessing():
    """Example 4: Advanced image preprocessing demonstration"""
    print("\n=== Example 4: Advanced Preprocessing ===")

    # Create a noisy image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background

    # Add some noise
    noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)

    # Draw a digit "7" manually with noise
    cv2.line(img, (20, 20), (80, 20), (0, 0, 0), 3)  # Top line
    cv2.line(img, (80, 20), (30, 80), (0, 0, 0), 3)  # Diagonal line

    # Save noisy image
    filename = "noisy_digit_7.png"
    cv2.imwrite(filename, img)

    print(f"Created noisy image: {filename}")

    # Test recognition
    digit, confidence = ai_recognize_digit(filename)

    if digit is not None:
        print(f"✓ AI handled noisy image: recognized {digit} (confidence: {confidence:.1f}%)")
    else:
        print(f"✗ AI struggled with noisy image (confidence: {confidence:.1f}%)")

    # Cleanup
    try:
        os.remove(filename)
    except:
        pass

def main():
    """Main function to run all examples"""
    print("🤖 AI Digit Recognition - Example Usage")
    print("=" * 50)

    # Check if AI is ready
    if ai_recognizer.model is None:
        print("❌ AI model not loaded. Please check your setup.")
        return

    print("✅ AI model loaded successfully!")

    # Run examples
    try:
        example_1_basic_recognition()

        # Ask user if they want to see GUI
        response = input("\nWould you like to launch the GUI application? (y/n): ").lower()
        if response in ['y', 'yes']:
            example_2_gui_application()

        example_3_integration_with_original()
        example_4_advanced_preprocessing()

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")

    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Try running 'python digit_recognition.py' for the full GUI")
    print("2. Use 'from utils import ai_recognize_digit' in your own scripts")
    print("3. Modify the confidence threshold in utils.py if needed")

if __name__ == "__main__":
    main()
