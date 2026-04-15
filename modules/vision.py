"""
OCR utilities for Dorm-Net.

This module improves image contrast before sending it to Tesseract so the OCR
result is more reliable, especially for classroom notes captured by phone.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytesseract


def extract_text(image_bytes: bytes) -> str:
    """
    Extract text from an image of handwritten notes.

    Steps used:
    1. Decode image bytes into an OpenCV image.
    2. Convert to grayscale.
    3. Apply thresholding to boost contrast.
    4. Run Tesseract OCR on the cleaned result.

    Args:
        image_bytes: Raw bytes from an uploaded image.

    Returns:
        A cleaned text string extracted from the image.
    """

    # Convert the byte stream to a NumPy array so OpenCV can decode it.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Uploaded image could not be decoded. Please try another file.")

    # Grayscale usually makes OCR preprocessing much easier.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A light blur helps reduce camera noise before thresholding.
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Adaptive thresholding works better than a fixed threshold when lighting
    # across the page is uneven.
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    # OCR configuration:
    # --oem 3 lets Tesseract choose the best OCR engine.
    # --psm 6 assumes a block of text, which fits class notes fairly well.
    custom_config = "--oem 3 --psm 6"
    extracted = pytesseract.image_to_string(thresholded, config=custom_config)

    # Clean up the output so the frontend displays something readable.
    cleaned_lines = [line.strip() for line in extracted.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)
