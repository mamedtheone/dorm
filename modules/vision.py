# vision_module.py
import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh

def extract_text(image_path):
    processed = preprocess_image(image_path)
    
    text = pytesseract.image_to_string(processed)
    
    return text

if __name__ == "__main__":
    text = extract_text("sample.jpg")
    print(text)