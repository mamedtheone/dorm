"""
Dorm-Net: Vision Module
Digitizes handwritten notes from images using OCR (Tesseract)
and adds them into the ChromaDB knowledge base.

Requirements:
    pip install pytesseract pillow opencv-python
    Also install Tesseract OCR engine:
    https://github.com/UB-Mannheim/tesseract/wiki  (Windows installer)
"""

import os
import uuid
import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    # If Tesseract is installed at a custom path on Windows, set it here:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not installed. Run: pip install pytesseract pillow opencv-python")

from brain_module import add_golden_summary

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Prepares an image for better OCR accuracy:
    - Converts to grayscale
    - Applies adaptive thresholding (handles uneven lighting)
    - Denoises the image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive threshold — great for handwritten notes on paper
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    
    return thresh

def extract_text_from_image(image_path: str) -> str:
    """
    Runs OCR on a preprocessed image and returns extracted text.
    """
    if not TESSERACT_AVAILABLE:
        return ""
    
    print(f"🔍 Processing image: {os.path.basename(image_path)}")
    
    processed = preprocess_image(image_path)
    pil_image = Image.fromarray(processed)
    
    # Use Tesseract with page segmentation mode 6 (single block of text)
    config = "--psm 6"
    text = pytesseract.image_to_string(pil_image, config=config)
    
    return text.strip()

def digitize_note(image_path: str, course: str, doc_id: str = None):
    """
    Full pipeline: image → OCR → ChromaDB
    """
    if not TESSERACT_AVAILABLE:
        print("❌ Tesseract not available. Please install it first.")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Extract text
    text = extract_text_from_image(image_path)
    
    if not text:
        print("⚠️  No text could be extracted from the image.")
        return
    
    print("\n📄 Extracted Text:")
    print("─" * 40)
    print(text)
    print("─" * 40)
    
    # Generate a unique ID if not provided
    if not doc_id:
        doc_id = f"note_{uuid.uuid4().hex[:8]}"
    
    # Save to ChromaDB
    add_golden_summary(
        text=text,
        document_id=doc_id,
        metadata={"course": course, "source": "handwritten_note", "image": os.path.basename(image_path)}
    )
    
    print(f"✅ Note digitized and saved as '{doc_id}' under course '{course}'!")
    return text

def digitize_folder(folder_path: str, course: str):
    """
    Processes all images in a folder and adds them to the knowledge base.
    """
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]
    
    if not images:
        print(f"No images found in {folder_path}")
        return
    
    print(f"📁 Found {len(images)} image(s) in '{folder_path}'. Processing...")
    
    for i, img_file in enumerate(images, 1):
        img_path = os.path.join(folder_path, img_file)
        doc_id = f"{course.lower().replace(' ', '_')}_{i:03d}"
        print(f"\n[{i}/{len(images)}] Processing: {img_file}")
        digitize_note(img_path, course=course, doc_id=doc_id)
    
    print(f"\n🎉 Done! {len(images)} note(s) added to the knowledge base.")

# ── Demo / Test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Dorm-Net: Vision Module ===")
    print()
    
    if not TESSERACT_AVAILABLE:
        print("To use this module, install the required packages:")
        print("  pip install pytesseract pillow opencv-python")
        print()
        print("Then install the Tesseract OCR engine (Windows):")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
    else:
        # Test with a specific image
        test_image = input("Enter path to a handwritten note image (or press Enter to skip): ").strip()
        if test_image:
            course = input("Course name: ").strip()
            digitize_note(test_image, course=course)
        else:
            print("Skipping demo. Import digitize_note() or digitize_folder() to use this module.")
