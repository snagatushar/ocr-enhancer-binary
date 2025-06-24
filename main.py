from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import re

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

# 🔁 Correct image orientation (90°, 180°, etc.)
def correct_orientation(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image)
        rotate_angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        orientation_conf = int(re.search(r"Orientation confidence: (\d+)", osd).group(1))

        print(f"[🔁] Rotation: {rotate_angle}° | Confidence: {orientation_conf}")
        if orientation_conf < 10 or rotate_angle == 0:
            return image
        return image.rotate(360 - rotate_angle, expand=True)
    except Exception as e:
        print(f"[❌] Orientation failed: {e}")
        return image

# 🧭 Deskew (slanted text fix)
def deskew_image(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return pil_img

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.5:
        return pil_img

    print(f"[🧭] Deskewing by {angle:.2f}°")
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(np.array(pil_img), M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(rotated)

# 🎨 Enhance image
def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)
    contrast = ImageEnhance.Contrast(gray).enhance(1.05)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.1)
    return sharpened

# ✅ Align only: rotate + deskew
@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    try:
        if not file:
            return JSONResponse(content={"error": "Image file is required."}, status_code=400)

        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        aligned = deskew_image(correct_orientation(image))
        img_bytes = BytesIO()
        aligned.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=aligned_image.png"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ Enhance only (for preview or saving)
@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        if not file:
            return JSONResponse(content={"error": "Image file is required."}, status_code=400)

        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        enhanced = enhance_image(deskew_image(correct_orientation(image)))
        img_bytes = BytesIO()
        enhanced.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=enhanced_output.png"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ OCR only (return text)
@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        if not file:
            return JSONResponse(content={"error": "Image file is required."}, status_code=400)

        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        processed = enhance_image(deskew_image(correct_orientation(image)))
        text = pytesseract.image_to_string(processed, config="--psm 6")

        return {"text": text.strip()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
