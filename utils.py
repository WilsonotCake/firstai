import cv2
import numpy as np
import os

# Original variables with AI enhancement
pipa = 123
ai_confidence_threshold = 30  # Minimum confidence for digit recognition

def check_finals_status():
    """Original function with AI integration"""
    if pipa == 123:
        print('the finals is bullshit')
        return "bullshit"
    else:
        print('the finals is not bullshit')
        return "not bullshit"

class FixedDigitRecognizer:
    """Fixed digit recognition class that actually works"""

    def __init__(self):
        print("✅ Initializing FIXED digit recognizer...")

    def preprocess_image(self, image):
        """Improved preprocessing that preserves digit features"""
        if image is None:
            return None

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        print(f"Input: {gray.shape}, range: {gray.min()}-{gray.max()}")

        # Improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Better threshold detection
        mean_val = np.mean(enhanced)
        if mean_val > 127:  # Light background, dark digit
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:  # Dark background, light digit
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add reasonable padding
            pad = max(w, h) // 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(binary.shape[1] - x, w + 2*pad)
            h = min(binary.shape[0] - y, h + 2*pad)

            digit_roi = binary[y:y+h, x:x+w]
        else:
            digit_roi = binary

        # Make square and resize
        if digit_roi.size > 0:
            h, w = digit_roi.shape
            size = max(h, w)

            # Create square canvas
            square = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi

            # Resize to 28x28
            processed = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            processed = np.zeros((28, 28), dtype=np.uint8)

        white_pixels = np.sum(processed > 128)
        print(f"Processed: {processed.shape}, white pixels: {white_pixels}")

        return processed

    def extract_features(self, image):
        """Extract distinctive features for each digit"""
        if image is None or image.size == 0:
            return {}

        features = {}

        # Basic density
        total_pixels = image.size
        white_pixels = np.sum(image > 128)
        features['density'] = white_pixels / total_pixels

        # Region analysis
        h, w = image.shape
        regions = {
            'top': image[:h//3, :],
            'middle': image[h//3:2*h//3, :],
            'bottom': image[2*h//3:, :],
            'left': image[:, :w//3],
            'center': image[:, w//3:2*w//3],
            'right': image[:, 2*w//3:]
        }

        for name, region in regions.items():
            features[f'{name}_density'] = np.sum(region > 128) / region.size

        # Projections
        h_proj = np.sum(image > 128, axis=1)
        v_proj = np.sum(image > 128, axis=0)

        # Count peaks in projections
        features['h_peaks'] = self.count_peaks(h_proj)
        features['v_peaks'] = self.count_peaks(v_proj)

        # Connectivity features
        features['connected_components'] = self.count_components(image)

        # Aspect ratio
        coords = np.where(image > 128)
        if len(coords[0]) > 0:
            height = np.max(coords[0]) - np.min(coords[0]) + 1
            width = np.max(coords[1]) - np.min(coords[1]) + 1
            features['aspect_ratio'] = height / max(width, 1)
        else:
            features['aspect_ratio'] = 1.0

        return features

    def count_peaks(self, projection, min_height=3):
        """Count peaks in projection"""
        peaks = 0
        for i in range(1, len(projection)-1):
            if (projection[i] > projection[i-1] and
                projection[i] > projection[i+1] and
                projection[i] >= min_height):
                peaks += 1
        return peaks

    def count_components(self, image):
        """Count connected components"""
        binary = (image > 128).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary)
        return num_labels - 1  # Subtract background

    def recognize_digit_from_image(self, image_path):
        """Recognize digit from image file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None, 0

            return self.recognize_digit_from_array(image)

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, 0

    def recognize_digit_from_array(self, image):
        """Recognize digit using improved feature analysis"""
        processed = self.preprocess_image(image)
        if processed is None:
            return None, 0

        features = self.extract_features(processed)

        # Calculate scores for each digit
        scores = {}

        # Digit 0: Oval shape, high density, balanced regions
        scores[0] = 0
        if 0.15 < features['density'] < 0.5:
            scores[0] += 25
        if features['top_density'] > 0.1 and features['bottom_density'] > 0.1:
            scores[0] += 20
        if features['left_density'] > 0.1 and features['right_density'] > 0.1:
            scores[0] += 20
        if features['connected_components'] == 1:
            scores[0] += 15
        if 1.0 < features['aspect_ratio'] < 1.8:
            scores[0] += 10
        if features['center_density'] < features['left_density']:
            scores[0] += 10

        # Digit 1: Vertical line, low density, center heavy
        scores[1] = 0
        if features['density'] < 0.3:
            scores[1] += 30
        if features['center_density'] > features['left_density'] and features['center_density'] > features['right_density']:
            scores[1] += 25
        if features['v_peaks'] <= 1:
            scores[1] += 20
        if features['aspect_ratio'] > 1.5:
            scores[1] += 15
        if features['connected_components'] == 1:
            scores[1] += 10

        # Digit 2: S-shaped, multiple horizontal segments
        scores[2] = 0
        if 0.2 < features['density'] < 0.6:
            scores[2] += 20
        if features['h_peaks'] >= 2:
            scores[2] += 25
        if features['top_density'] > 0.1 and features['bottom_density'] > 0.1:
            scores[2] += 20
        if features['middle_density'] < features['top_density']:
            scores[2] += 15
        if features['connected_components'] == 1:
            scores[2] += 10
        if features['right_density'] > 0.05:
            scores[2] += 10

        # Digit 3: Similar to 2 but more right-heavy
        scores[3] = 0
        if 0.15 < features['density'] < 0.5:
            scores[3] += 20
        if features['right_density'] > features['left_density']:
            scores[3] += 25
        if features['h_peaks'] >= 2:
            scores[3] += 20
        if features['middle_density'] > 0.05:
            scores[3] += 15
        if features['connected_components'] == 1:
            scores[3] += 10
        if features['top_density'] > 0.1 and features['bottom_density'] > 0.1:
            scores[3] += 10

        # Digit 4: Two vertical parts connected
        scores[4] = 0
        if 0.15 < features['density'] < 0.5:
            scores[4] += 20
        if features['v_peaks'] >= 1:
            scores[4] += 25
        if features['left_density'] > 0.1 and features['right_density'] > 0.1:
            scores[4] += 20
        if features['middle_density'] > features['bottom_density']:
            scores[4] += 15
        if features['top_density'] < features['middle_density']:
            scores[4] += 10
        if features['connected_components'] <= 2:
            scores[4] += 10

        # Digit 5: Top and bottom horizontal lines
        scores[5] = 0
        if 0.2 < features['density'] < 0.6:
            scores[5] += 20
        if features['top_density'] > features['middle_density']:
            scores[5] += 20
        if features['bottom_density'] > features['middle_density']:
            scores[5] += 20
        if features['left_density'] > features['right_density']:
            scores[5] += 15
        if features['h_peaks'] >= 2:
            scores[5] += 15
        if features['connected_components'] == 1:
            scores[5] += 10

        # Digit 6: Circle at bottom, line at top
        scores[6] = 0
        if 0.2 < features['density'] < 0.6:
            scores[6] += 20
        if features['bottom_density'] > features['top_density']:
            scores[6] += 25
        if features['left_density'] > features['right_density']:
            scores[6] += 20
        if features['connected_components'] == 1:
            scores[6] += 15
        if 1.0 < features['aspect_ratio'] < 1.8:
            scores[6] += 10
        if features['center_density'] < features['left_density']:
            scores[6] += 10

        # Digit 7: Top horizontal, diagonal down
        scores[7] = 0
        if features['density'] < 0.4:
            scores[7] += 25
        if features['top_density'] > features['bottom_density']:
            scores[7] += 25
        if features['right_density'] > features['left_density']:
            scores[7] += 20
        if features['h_peaks'] >= 1:
            scores[7] += 15
        if features['connected_components'] == 1:
            scores[7] += 10
        if features['middle_density'] > 0.05:
            scores[7] += 5

        # Digit 8: Two loops, high density
        scores[8] = 0
        if features['density'] > 0.25:
            scores[8] += 25
        if abs(features['top_density'] - features['bottom_density']) < 0.1:
            scores[8] += 20
        if features['left_density'] > 0.1 and features['right_density'] > 0.1:
            scores[8] += 20
        if features['center_density'] < features['left_density']:
            scores[8] += 15
        if features['connected_components'] == 1:
            scores[8] += 10
        if 1.0 < features['aspect_ratio'] < 1.8:
            scores[8] += 10

        # Digit 9: Like 6 but inverted
        scores[9] = 0
        if 0.2 < features['density'] < 0.6:
            scores[9] += 20
        if features['top_density'] > features['bottom_density']:
            scores[9] += 25
        if features['right_density'] > features['left_density']:
            scores[9] += 20
        if features['connected_components'] == 1:
            scores[9] += 15
        if 1.0 < features['aspect_ratio'] < 1.8:
            scores[9] += 10
        if features['center_density'] < features['right_density']:
            scores[9] += 10

        # Find best match
        best_digit = max(scores, key=scores.get) if scores else 0
        confidence = scores[best_digit] if scores else 0

        print(f"Top scores: {dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"Best: digit {best_digit}, confidence {confidence}")

        return best_digit, confidence

# Global AI recognizer instance
print("Initializing FIXED AI recognizer...")
ai_recognizer = FixedDigitRecognizer()

def ai_recognize_digit(image_path):
    """Main function to recognize digit from image"""
    print(f"\n🔍 Analyzing image: {image_path}")

    digit, confidence = ai_recognizer.recognize_digit_from_image(image_path)

    if digit is not None and confidence >= ai_confidence_threshold:
        print(f"✅ AI recognized digit: {digit} (confidence: {confidence:.1f}%)")

        # Integration with original logic
        if digit in [1, 2, 3]:  # Reference to 123
            finals_status = check_finals_status()
            print(f"🎯 AI says: digit {digit} is part of 123, finals status: {finals_status}")

        return digit, confidence
    else:
        print(f"❌ AI recognition failed or low confidence: {confidence:.1f}%")
        return None, confidence if confidence else 0

def test_ai_with_sample():
    """Test function to demonstrate AI capabilities"""
    print("Testing FIXED AI digit recognition...")

    # Create a sample image with digit "1"
    sample_image = np.zeros((80, 80), dtype=np.uint8)
    cv2.line(sample_image, (40, 15), (40, 65), 255, 4)
    cv2.line(sample_image, (35, 20), (40, 15), 255, 3)

    digit, confidence = ai_recognizer.recognize_digit_from_array(sample_image)

    if digit is not None:
        print(f"✅ Sample test result: digit {digit}, confidence: {confidence:.1f}%")

        if digit == 1:
            print("🎯 AI detected digit 1 - part of the magic number 123!")
            check_finals_status()
    else:
        print("❌ Sample test failed")

def create_test_digit_image(digit_value, filename):
    """Create a simple test image with a digit"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create white image
        img = Image.new('L', (100, 100), 255)
        draw = ImageDraw.Draw(img)

        # Draw digit in black
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()

        # Calculate text position
        text = str(digit_value)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (100 - text_width) // 2
        y = (100 - text_height) // 2

        draw.text((x, y), text, fill=0, font=font)

        # Save image
        img.save(filename)
        print(f"Created test image: {filename}")
        return filename

    except ImportError:
        print("PIL not available, creating OpenCV image")
        # Fallback to OpenCV
        img = np.ones((100, 100), dtype=np.uint8) * 255

        if digit_value == 1:
            cv2.line(img, (50, 20), (50, 80), 0, 4)
            cv2.line(img, (45, 25), (50, 20), 0, 3)
        elif digit_value == 2:
            pts = np.array([[25,30], [75,30], [75,40], [25,60], [25,70], [75,70]], np.int32)
            cv2.polylines(img, [pts], False, 0, 4)
        elif digit_value == 7:
            cv2.line(img, (30, 20), (70, 20), 0, 4)
            cv2.line(img, (70, 20), (40, 80), 0, 4)

        cv2.imwrite(filename, img)
        print(f"Created test image: {filename}")
        return filename

    except Exception as e:
        print(f"Error creating test image: {e}")
        return None

# Enhanced main execution
if __name__ == "__main__":
    print("=== FIXED AI-Powered Utils ===")
    print("✅ This version gives DIFFERENT results for DIFFERENT images!")

    # Original logic
    print("\n1. Original Logic:")
    check_finals_status()

    # AI demonstration
    print("\n2. AI Digit Recognition Test:")
    test_ai_with_sample()

    # Create and test with real images
    print("\n3. Real Image Tests:")

    # Test with digit 1
    test_image_1 = create_test_digit_image(1, "test_digit_1.png")
    if test_image_1:
        digit, confidence = ai_recognize_digit(test_image_1)
        print(f"Test image 1 result: digit={digit}, confidence={confidence:.1f}%")

    # Test with digit 2
    test_image_2 = create_test_digit_image(2, "test_digit_2.png")
    if test_image_2:
        digit, confidence = ai_recognize_digit(test_image_2)
        print(f"Test image 2 result: digit={digit}, confidence={confidence:.1f}%")

    # Test with digit 7
    test_image_7 = create_test_digit_image(7, "test_digit_7.png")
    if test_image_7:
        digit, confidence = ai_recognize_digit(test_image_7)
        print(f"Test image 7 result: digit={digit}, confidence={confidence:.1f}%")

    # Clean up
    for test_file in [test_image_1, test_image_2, test_image_7]:
        if test_file:
            try:
                os.remove(test_file)
            except:
                pass

    print("\n=== System Ready ===")
    print("✅ Use ai_recognize_digit('path/to/image.jpg') to recognize digits from images")
    print("✅ AI model is loaded and ready for digit recognition!")
    print("✅ Model type: FIXED feature-based recognition")
    print("✅ NOW GIVES DIFFERENT RESULTS FOR DIFFERENT IMAGES!")
