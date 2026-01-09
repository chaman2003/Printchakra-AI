"""
OCR & AI Recognition Module - Improved multi-config OCR extraction
Text extraction with multiple preprocessing variants and configurations
Based on printchakra_clean.ipynb Section 5
Includes PaddleOCR for advanced OCR with bounding boxes
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Import PaddleOCR components
try:
    from .paddle_ocr import (
        PaddleOCRProcessor,
        OCRResult,
        get_ocr_processor,
        get_paddle_ocr,
    )
    PADDLE_OCR_AVAILABLE = True
except ImportError as e:
    PADDLE_OCR_AVAILABLE = False
    print(f"⚠️  PaddleOCR not available: {e}")


class OCRModule:
    """
    Handles OCR text extraction with multi-configuration approach
    Based on printchakra_clean.ipynb Section 5
    """

    def __init__(self, language: str = "eng", psm: int = 3, oem: int = 3):
        """
        Initialize OCR module

        Args:
            language: Tesseract language code (e.g., 'eng', 'eng+fra')
            psm: Page segmentation mode (0-13)
            oem: OCR Engine mode (0-3)
                0 = Legacy engine only
                1 = Neural nets LSTM engine only
                2 = Legacy + LSTM engines
                3 = Default (based on what is available)
        """
        self.language = language
        self.psm = psm
        self.oem = oem
        self.config = f"--oem {oem} --psm {psm}"

    def extract_text_multi_config(self, image: np.ndarray, debug: bool = False) -> Tuple[str, Dict]:
        """
        Extract text with multiple OCR configurations and preprocessing variants
        From notebook Section 5 - improved multi-config approach

        Args:
            image: Input image
            debug: Print debug information

        Returns:
            Tuple of (best_text, statistics_dict)
        """
        from app.modules.image.enhancement import ImageEnhancer

        configs = [
            ("PSM 3 (Auto)", "--oem 3 --psm 3"),
            ("PSM 6 (Block)", "--oem 3 --psm 6"),
            ("PSM 4 (Column)", "--oem 3 --psm 4"),
        ]

        # Get preprocessing variants
        enhancer = ImageEnhancer()
        variants = enhancer.preprocess_for_ocr(image)
        results = []

        try:
            for img, variant_name in variants:
                for config_name, config_str in configs:
                    text = pytesseract.image_to_string(img, lang=self.language, config=config_str)
                    results.append(
                        {
                            "text": text,
                            "length": len(text.strip()),
                            "config": config_name,
                            "variant": variant_name,
                        }
                    )

            # Select best result (longest text)
            best = max(results, key=lambda x: x["length"])

            if debug:
                print(f"✅ Best OCR: {best['config']} ({best['variant']}) - {best['length']} chars")

            # Get detailed stats from first variant
            data = pytesseract.image_to_data(variants[0][0], output_type=pytesseract.Output.DICT)
            words = [w for w in data["text"] if w.strip()]
            lines = set(data["line_num"])

            stats = {
                "chars": len(best["text"]),
                "words": len(words),
                "lines": len(lines),
                "config": best["config"],
                "variant": best["variant"],
            }

            return best["text"], stats

        except Exception as e:
            if debug:
                print(f"OCR error: {e}")
            return "", {}

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from image (simple version for compatibility)

        Args:
            image: Input image

        Returns:
            Extracted text
        """
        try:
            text = pytesseract.image_to_string(image, lang=self.language, config=self.config)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""

    def extract_text_with_confidence(self, image: np.ndarray) -> Dict:
        """
        Extract text with confidence scores using multi-config approach

        Args:
            image: Input image

        Returns:
            Dict with text and confidence data
        """
        try:
            # Use multi-config extraction for better results
            text, stats = self.extract_text_multi_config(image, debug=False)

            # Get confidence data
            data = pytesseract.image_to_data(
                image, lang=self.language, config=self.config, output_type=pytesseract.Output.DICT
            )

            # Filter out low confidence results
            filtered_text = []
            confidences = []

            for i in range(len(data["text"])):
                if int(data["conf"][i]) > 30:  # Confidence threshold
                    text_word = data["text"][i].strip()
                    if text_word:
                        filtered_text.append(text_word)
                        confidences.append(int(data["conf"][i]))

            avg_confidence = np.mean(confidences) if confidences else 0

            return {
                "text": text,  # Use multi-config text (better quality)
                "confidence": float(avg_confidence),
                "word_count": stats.get("words", len(filtered_text)),
                "all_confidences": confidences,
            }
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return {"text": "", "confidence": 0, "word_count": 0}

    def extract_structured_data(self, image: np.ndarray) -> List[Dict]:
        """
        Extract structured data (boxes with text and positions)

        Args:
            image: Input image

        Returns:
            List of dicts with text and bounding box info
        """
        try:
            data = pytesseract.image_to_data(
                image, lang=self.language, config=self.config, output_type=pytesseract.Output.DICT
            )

            structured = []

            for i in range(len(data["text"])):
                if int(data["conf"][i]) > 30:
                    text = data["text"][i].strip()
                    if text:
                        structured.append(
                            {
                                "text": text,
                                "confidence": int(data["conf"][i]),
                                "x": data["left"][i],
                                "y": data["top"][i],
                                "width": data["width"][i],
                                "height": data["height"][i],
                                "block_num": data["block_num"][i],
                                "line_num": data["line_num"][i],
                            }
                        )

            return structured
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return []


class DocumentClassifier:
    """
    Classify document types using KNN classifier
    """

    # Document types
    DOCUMENT_TYPES = ["ID", "BILL", "RECEIPT", "FORM", "NOTE", "OTHER"]

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize document classifier

        Args:
            model_path: Path to saved model (optional)
        """
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.scaler = StandardScaler()
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image for classification

        Args:
            image: Input image

        Returns:
            Feature vector
        """
        try:
            # Convert to grayscale safely
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Feature 1: Aspect ratio
            h, w = gray.shape
            aspect_ratio = w / h if h > 0 else 1.0

            # Feature 2: Text density (approximate)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            text_density = np.sum(binary > 0) / binary.size

            # Feature 3: Edge density using Canny (matching notebook thresholds)
            try:
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
            except Exception:
                edges = np.zeros_like(gray)
                edge_density = 0.0

            # Feature 4: Horizontal lines (common in bills/receipts)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10
            )
            horizontal_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if angle < 10 or angle > 170:  # Nearly horizontal
                        horizontal_lines += 1

            # Feature 5: Vertical lines (common in forms)
            vertical_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if 80 < angle < 100:  # Nearly vertical
                        vertical_lines += 1

            # Feature 6: Mean intensity
            mean_intensity = np.mean(gray)

            # Feature 7: Std deviation of intensity
            std_intensity = np.std(gray)

            # Feature 8: Contour count (complexity)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)

            # Combine features
            features = np.array(
                [
                    aspect_ratio,
                    text_density,
                    edge_density,
                    horizontal_lines,
                    vertical_lines,
                    mean_intensity,
                    std_intensity,
                    contour_count,
                ]
            )

            return features

        except Exception as e:
            # Return default features on error
            return np.zeros(8)

    def train(self, images: List[np.ndarray], labels: List[str]):
        """
        Train the classifier

        Args:
            images: List of training images
            labels: List of corresponding labels
        """
        # Extract features
        features = np.array([self.extract_features(img) for img in images])

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train model
        self.model.fit(features_scaled, labels)
        self.is_trained = True

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict document type

        Args:
            image: Input image

        Returns:
            Tuple of (predicted_type, confidence)
        """
        if not self.is_trained:
            return "OTHER", 0.0

        # Extract and scale features
        features = self.extract_features(image).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]

        # Get probability/confidence
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))

        return prediction, confidence

    def save_model(self, path: str):
        """Save trained model"""
        model_data = {"model": self.model, "scaler": self.scaler, "is_trained": self.is_trained}
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str):
        """Load trained model"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = model_data["is_trained"]


class AIEnhancer:
    """
    AI-based image enhancement (placeholder for future ML models)
    """

    def __init__(self):
        """Initialize AI enhancer"""
        # Placeholder for future MobileNet or custom CNN integration
        self.model = None

    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using AI

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        # For now, use traditional enhancement
        # TODO: Integrate MobileNet or custom trained model

        # Convert to LAB color space
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def super_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        Apply super-resolution (placeholder)

        Args:
            image: Input image

        Returns:
            High-resolution image
        """
        # Placeholder for future DNN super-resolution
        # For now, use bicubic interpolation
        scale_factor = 2
        height, width = image.shape[:2]
        enhanced = cv2.resize(
            image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC
        )
        return enhanced


print("[OK] OCR module loaded (improved multi-config)")

# Global OCR service instance for convenience
ocr_service = OCRModule()

__all__ = [
    'OCRModule',
    'ocr_service',
    'PADDLE_OCR_AVAILABLE',
]

if PADDLE_OCR_AVAILABLE:
    __all__.extend(['PaddleOCRProcessor', 'OCRResult', 'get_ocr_processor', 'get_paddle_ocr'])
