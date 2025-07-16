#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Test Images for Digit Recognition
Simple script to generate test images with clear digits
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_digit_cv2(digit, size=(100, 100)):
    """Create a simple digit image using OpenCV"""
    img = np.zeros(size, dtype=np.uint8)

    # Define digit patterns
    if digit == 0:
        cv2.ellipse(img, (50, 50), (30, 40), 0, 0, 360, 255, 3)
    elif digit == 1:
        cv2.line(img, (50, 20), (50, 80), 255, 3)
        cv2.line(img, (45, 25), (50, 20), 255, 2)
        cv2.line(img, (30, 80), (70, 80), 255, 2)
    elif digit == 2:
        cv2.line(img, (30, 30), (70, 30), 255, 3)
        cv2.line(img, (70, 30), (70, 50), 255, 3)
        cv2.line(img, (70, 50), (30, 70), 255, 3)
        cv2.line(img, (30, 70), (30, 80), 255, 3)
        cv2.line(img, (30, 80), (70, 80), 255, 3)
    elif digit == 3:
        cv2.line(img, (30, 30), (70, 30), 255, 3)
        cv2.line(img, (70, 30), (70, 50), 255, 3)
        cv2.line(img, (50, 50), (70, 50), 255, 3)
        cv2.line(img, (70, 50), (70, 70), 255, 3)
        cv2.line(img, (70, 70), (30, 70), 255, 3)
    elif digit == 4:
        cv2.line(img, (30, 20), (30, 55), 255, 3)
        cv2.line(img, (30, 55), (60, 55), 255, 3)
        cv2.line(img, (60, 20), (60, 80), 255, 3)
    elif digit == 5:
        cv2.line(img, (30, 20), (70, 20), 255, 3)
        cv2.line(img, (30, 20), (30, 50), 255, 3)
        cv2.line(img, (30, 50), (60, 50), 255, 3)
        cv2.line(img, (60, 50), (60, 70), 255, 3)
        cv2.line(img, (60, 70), (30, 70), 255, 3)
    elif digit == 6:
        cv2.ellipse(img, (50, 60), (20, 15), 0, 0, 360, 255, 3)
        cv2.line(img, (30, 20), (30, 60), 255, 3)
        cv2.line(img, (30, 20), (60, 20), 255, 3)
    elif digit == 7:
        cv2.line(img, (30, 20), (70, 20), 255, 3)
        cv2.line(img, (70, 20), (40, 80), 255, 3)
    elif digit == 8:
        cv2.ellipse(img, (50, 35), (15, 12), 0, 0, 360, 255, 3)
        cv2.ellipse(img, (50, 65), (18, 15), 0, 0, 360, 255, 3)
    elif digit == 9:
        cv2.ellipse(img, (50, 40), (20, 15), 0, 0, 360, 255, 3)
        cv2.line(img, (70, 40), (70, 80), 255, 3)
        cv2.line(img, (70, 80), (40, 80), 255, 3)

    return img

def create_digit_with_pil(digit, size=(100, 100)):
    """Create digit image using PIL with text"""
    try:
        # Create white background
        img = Image.new('L', size, 255)
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
            except:
                font = ImageFont.load_default()

        # Calculate text position to center it
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        # Draw black text
        draw.text((x, y), text, fill=0, font=font)

        return np.array(img)

    except Exception as e:
        print(f"PIL method failed: {e}, using OpenCV method")
        return create_simple_digit_cv2(digit, size)

def create_noisy_digit(digit, size=(100, 100)):
    """Create a digit with some noise"""
    img = create_digit_with_pil(digit, size)

    # Add some noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add slight blur
    img_blurred = cv2.GaussianBlur(img_noisy, (3, 3), 0.5)

    return img_blurred

def create_rotated_digit(digit, size=(100, 100), angle=15):
    """Create a slightly rotated digit"""
    img = create_digit_with_pil(digit, size)

    # Get rotation matrix
    center = (size[0]//2, size[1]//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    rotated = cv2.warpAffine(img, rotation_matrix, size,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)

    return rotated

def create_all_test_images():
    """Create test images for all digits with variations"""

    # Create output directory
    output_dir = "test_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Creating test images...")

    for digit in range(10):
        # Clear digit
        clear_img = create_digit_with_pil(digit)
        cv2.imwrite(f"{output_dir}/digit_{digit}_clear.png", clear_img)

        # OpenCV version
        cv_img = create_simple_digit_cv2(digit)
        cv2.imwrite(f"{output_dir}/digit_{digit}_cv.png", cv_img)

        # Noisy version
        noisy_img = create_noisy_digit(digit)
        cv2.imwrite(f"{output_dir}/digit_{digit}_noisy.png", noisy_img)

        # Rotated version
        rotated_img = create_rotated_digit(digit, angle=10)
        cv2.imwrite(f"{output_dir}/digit_{digit}_rotated.png", rotated_img)

        print(f"✓ Created images for digit {digit}")

    print(f"\n🎉 All test images created in '{output_dir}' folder!")
    print(f"Total images: {len(os.listdir(output_dir))}")

    return output_dir

def test_single_digit(digit, show_preview=True):
    """Create and optionally display a single digit"""
    print(f"Creating test image for digit {digit}...")

    # Create the image
    img = create_digit_with_pil(digit)
    filename = f"test_digit_{digit}.png"
    cv2.imwrite(filename, img)

    print(f"✓ Saved as {filename}")

    if show_preview:
        # Show image info
        print(f"Image size: {img.shape}")
        print(f"Image type: {img.dtype}")
        print(f"Pixel range: {img.min()} - {img.max()}")

    return filename

def main():
    """Main function to demonstrate test image creation"""
    print("🖼️  Test Image Creator for Digit Recognition")
    print("=" * 50)

    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Create all test images (0-9 with variations)")
    print("2. Create single digit image")
    print("3. Quick test with digit 1")

    try:
        choice = input("\nEnter choice (1-3) or press Enter for quick test: ").strip()

        if choice == "1":
            create_all_test_images()

        elif choice == "2":
            digit = int(input("Enter digit (0-9): "))
            if 0 <= digit <= 9:
                filename = test_single_digit(digit)
                print(f"\n🎯 You can now test this image with:")
                print(f"   from utils import ai_recognize_digit")
                print(f"   ai_recognize_digit('{filename}')")
            else:
                print("❌ Invalid digit! Must be 0-9")

        else:  # Default or choice 3
            print("Creating quick test image for digit 1...")
            filename = test_single_digit(1)

            # Try to test it immediately if utils is available
            try:
                from utils import ai_recognize_digit
                print("\n🤖 Testing with AI...")
                digit, confidence = ai_recognize_digit(filename)
                print(f"Result: digit={digit}, confidence={confidence:.1f}%")

                # Clean up
                os.remove(filename)
                print(f"Cleaned up {filename}")

            except ImportError:
                print(f"\n💡 To test this image, run:")
                print(f"   python -c \"from utils import ai_recognize_digit; print(ai_recognize_digit('{filename}'))\"")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except ValueError:
        print("❌ Invalid input!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
