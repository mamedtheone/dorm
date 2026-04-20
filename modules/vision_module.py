"""
vision_module.py — Dorm-Net Vision Engine (Enhanced)
====================================================
Upgraded with contrast normalization + safer OCR pipeline.

Pipeline:
Load → Denoise → Grayscale → Contrast Boost → Deskew → Threshold → Edge-enhance → OCR
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
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
    TESS_CONFIG = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?@#$%^&*()+-=/\\ "

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._verify_tesseract()

    # ---------------------- Public API ---------------------- #

    async def extract_text_async(self, image_input) -> OCRResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.extract_text, image_input)

    def extract_text(self, image_input) -> OCRResult:
        steps: list[str] = []
        try:
            img = self._load_image(image_input)
            steps.append("load")

            img = self._denoise(img)
            steps.append("denoise")

            gray = self._to_grayscale(img)
            steps.append("grayscale")

            gray = self._boost_contrast(gray)
            steps.append("contrast_boost")

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

    # ---------------------- Processing ---------------------- #

    def _load_image(self, source) -> np.ndarray:
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
        return cv2.fastNlMeansDenoisingColored(
            img, None, h=10, hColor=10,
            templateWindowSize=7, searchWindowSize=21
        )

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _boost_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Great for faint pencil or uneven lighting.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            return gray

        angles = []
        for line in lines[:50]:
            _, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return gray

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)

        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    def _edge_enhance(self, binary: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(binary, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, 1)
        return cv2.subtract(binary, edges)

    def _run_ocr(self, processed: np.ndarray) -> tuple[str, float]:
        pil_img = Image.fromarray(processed)

        data = pytesseract.image_to_data(
            pil_img,
            config=self.TESS_CONFIG,
            output_type=pytesseract.Output.DICT
        )

        confs = [
            int(c) for c in data["conf"]
            if str(c).lstrip("-").isdigit() and int(c) > 30
        ]

        confidence = float(np.mean(confs)) if confs else 0.0

        text = pytesseract.image_to_string(
            pil_img,
            config=self.TESS_CONFIG
        )

        return text, confidence

    # ---------------------- Utils ---------------------- #

    def _verify_tesseract(self):
        try:
            ver = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {ver}")
        except Exception as exc:
            logger.warning(f"Tesseract not found: {exc}")