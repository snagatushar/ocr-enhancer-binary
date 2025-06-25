from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import os

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()


def smart_deskew(pil_img: Image.Image, angle_threshold: float = 2.0) -> Image.Image:
    # Convert to grayscale for processing
    gray = np.array(pil_img.convert("L"))
    
    # Apply adaptive thresholding to better handle varying document backgrounds
    # This helps with documents that have watermarks or colored backgrounds
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to detect text lines which help determine orientation
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours that might be noise
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    
    if not contours:
        print("[ℹ️] No significant contours found, skipping deskew.")
        return pil_img
    
    # Get all non-zero points for angle detection
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        print("[ℹ️] No foreground pixels found, skipping deskew.")
        return pil_img

    # Calculate the minimum area rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Adjust angle based on orientation
    # This improved calculation handles both portrait and landscape documents
    if rect[1][0] > rect[1][1]:  # width > height, landscape orientation
        angle = -(90 + angle) if angle < -45 else -angle
    else:  # portrait orientation
        angle = -(90 + angle) if angle < -45 else -angle
    
    # Skip small angles to avoid unnecessary rotations
    if abs(angle) <= angle_threshold:
        print(f"[✅] Angle {angle:.2f}° is within threshold ({angle_threshold}°), skipping deskew.")
        return pil_img

    print(f"[🧭] Deskewing... angle detected: {angle:.2f}°")

    # Get image dimensions
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new dimensions to ensure the entire image fits
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the translation component of the matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Apply the rotation
    rotated = cv2.warpAffine(np.array(pil_img), M, (new_w, new_h), 
                            flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    
    return Image.fromarray(rotated)


def enhance_image(image: Image.Image) -> Image.Image:
    print("[✨] Enhancing image...")
    gray = ImageOps.grayscale(image)

    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    # Increase contrast slightly more for better text visibility
    contrast = ImageEnhance.Contrast(gray).enhance(1.2)
    # Increase sharpness for clearer text edges
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.3)
    # Add brightness adjustment for better readability
    brightened = ImageEnhance.Brightness(sharpened).enhance(1.1)

    return brightened

# ... existing code ...