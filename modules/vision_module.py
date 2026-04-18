"""
vision_module.py — Dorm-Net Vision Engine
==========================================
Handles image preprocessing and OCR extraction for handwritten notes.
Pipeline: Load → Denoise → Grayscale → Deskew → Threshold → Edge-enhance → OCR
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class OCRResult:
    raw_text: str
    confidence: float
    preprocessing_steps: list[str]
    word_count: int
    success: bool
    error: Optional[str] = None


class VisionEngine:
    """
    Robust OCR engine with multi-stage image preprocessing.

    Optimised for:
      - Messy handwritten engineering notes
      - Low-contrast scans / phone photos
      - Skewed or rotated pages
    """

    # Tesseract config: treat image as full page, allow digits + letters
    TESS_CONFIG = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?@#$%^&*()+-=/\\ "

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Args:
            tesseract_cmd: Path to tesseract binary.
                           E.g. r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe" on Windows.
                           Leave None to use system PATH.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._verify_tesseract()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def extract_text_async(self, image_input) -> OCRResult:
        """
        Non-blocking OCR. Runs the CPU-heavy work in a thread pool so
        Streamlit's event loop stays responsive.

        Args:
            image_input: bytes | np.ndarray | PIL.Image | str (file path)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.extract_text, image_input)

    def extract_text(self, image_input) -> OCRResult:
        """
        Synchronous OCR pipeline.
        """
        steps: list[str] = []
        try:
            img = self._load_image(image_input)
            steps.append("load")

            img = self._denoise(img)
            steps.append("denoise")

            gray = self._to_grayscale(img)
            steps.append("grayscale")

            gray = self._deskew(gray)
            steps.append("deskew")

            processed = self._adaptive_threshold(gray)
            steps.append("adaptive_threshold")

            processed = self._edge_enhance(processed)
            steps.append("edge_enhance")

            text, confidence = self._run_ocr(processed)
            steps.append("ocr")

            return OCRResult(
                raw_text=text.strip(),
                confidence=confidence,
                preprocessing_steps=steps,
                word_count=len(text.split()),
                success=True,
            )

        except Exception as exc:
            logger.exception("OCR pipeline failed")
            return OCRResult(
                raw_text="",
                confidence=0.0,
                preprocessing_steps=steps,
                word_count=0,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------ #
    #  Preprocessing stages                                                #
    # ------------------------------------------------------------------ #

    def _load_image(self, source) -> np.ndarray:
        """Accept bytes, PIL Image, ndarray, or file path."""
        if isinstance(source, np.ndarray):
            return source.copy()
        if isinstance(source, Image.Image):
            return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
        if isinstance(source, (bytes, bytearray)):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        if isinstance(source, str):
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Image not found: {source}")
            return img
        raise TypeError(f"Unsupported image type: {type(source)}")

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Non-local Means Denoising — preserves edges better than Gaussian blur
        for handwritten ink on paper.
        """
        return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                               templateWindowSize=7, searchWindowSize=21)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Estimate skew angle using Hough Line Transform and rotate to correct it.
        Handles up to ±45° tilt — common when photographing notes at an angle.
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return gray  # No lines detected; skip rotation

        angles = []
        for line in lines[:min(len(lines), 50)]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return gray  # Close enough to straight

        (h, w) = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug(f"Deskewed by {median_angle:.2f}°")
        return rotated

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Gaussian Adaptive Thresholding — handles uneven lighting across the page
        (e.g. shadow on one side of a notebook).

        Block size 11 and C=2 work well for A4-sized text at 150–300 dpi.
        """
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    def _edge_enhance(self, binary: np.ndarray) -> np.ndarray:
        """
        Canny Edge Detection overlay to sharpen character boundaries.
        Dilate edges slightly before merging to thicken thin handwriting strokes.
        """
        edges = cv2.Canny(binary, threshold1=30, threshold2=100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # Subtract edges from binary to sharpen (invert logic: 0 = ink, 255 = paper)
        enhanced = cv2.subtract(binary, dilated_edges)
        return enhanced

    def _run_ocr(self, processed: np.ndarray) -> tuple[str, float]:
        """
        Run Tesseract and compute a mean confidence score.
        Returns (text, confidence_0_to_100).
        """
        pil_img = Image.fromarray(processed)

        # Full data for confidence metrics
        data = pytesseract.image_to_data(
            pil_img,
            config=self.TESS_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

        # Filter words with detectable confidence
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        mean_conf = float(np.mean(confs)) if confs else 0.0

        text = pytesseract.image_to_string(pil_img, config=self.TESS_CONFIG)
        return text, mean_conf

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _verify_tesseract(self):
        try:
            ver = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {ver}")
        except Exception as exc:
            logger.warning(
                f"Tesseract not found on PATH: {exc}\n"
                "Set tesseract_cmd in VisionEngine() or install Tesseract."
            )

    def image_to_pil(self, img: np.ndarray) -> Image.Image:
        """Helper to convert processed ndarray back to PIL for display."""
        if len(img.shape) == 2:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
