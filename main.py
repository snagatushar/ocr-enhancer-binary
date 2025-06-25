from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import os
import re

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

def smart_deskew(pil_img: Image.Image, angle_threshold: float = 2.0) -> Image.Image:
    print("[🔍] Trying Hough Transform first...")
    gray = np.array(pil_img.convert("L"))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angles = [np.rad2deg(theta) - 90 for rho, theta in lines[:, 0]]
        median_angle = np.median(angles)

        if abs(median_angle) > angle_threshold:
            print(f"[🧭] Hough deskew angle: {-median_angle:.2f}°")
            (h, w) = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -median_angle, 1.0)
            rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            return Image.fromarray(rotated)
        else:
            print(f"[✅] Hough angle {median_angle:.2f}° within threshold. No rotation.")
            return pil_img

    print("[⚠️] Hough failed. Falling back to Tesseract OSD...")
    try:
        osd = pytesseract.image_to_osd(pil_img)
        rotate_angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        print(f"[🔁] Tesseract OSD angle: {rotate_angle}°")
        if rotate_angle != 0:
            return pil_img.rotate(360 - rotate_angle, expand=True)
    except Exception as e:
        print(f"[❌] Tesseract OSD failed: {e}")

    return pil_img


def enhance_image(image: Image.Image) -> Image.Image:
    print("[✨] Enhancing image: contrast & sharpness")
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
