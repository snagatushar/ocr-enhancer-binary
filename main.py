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
    print("[🔍] Deskewing with Hough Transform...")

    gray = np.array(pil_img.convert("L"))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        print("[ℹ️] No lines detected. Skipping deskew.")
        return pil_img

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.rad2deg(theta) - 90
        angles.append(angle)

    median_angle = np.median(angles)

    if abs(median_angle) <= angle_threshold:
        print(f"[✅] Median angle {median_angle:.2f}° within threshold. Skipping rotation.")
        return pil_img

    print(f"[🧭] Deskewing... Rotating by {-median_angle:.2f}°")

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -median_angle, 1.0)

    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(rotated)


def enhance_image(image: Image.Image) -> Image.Image:
    print("[✨] Enhancing (contrast, sharpness). No rotation.")
    gray = ImageOps.grayscale(image)

    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    contrast = ImageEnhance.Contrast(gray).enhance(1.05)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.1)

    return sharpened


@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        aligned = smart_deskew(image)

        img_bytes = BytesIO()
        aligned.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=aligned_image.png"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        aligned = smart_deskew(image)
        enhanced = enhance_image(aligned)

        img_bytes = BytesIO()
        enhanced.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=enhanced_output.png"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        aligned = smart_deskew(image)
        enhanced = enhance_image(aligned)

        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {"status": "ok"}
