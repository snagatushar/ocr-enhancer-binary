from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import re

# For Docker/Render: set tesseract path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

def correct_orientation(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image)
        rotate_angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        print(f"[🔁] Tesseract auto-rotation: {rotate_angle}°")
        if rotate_angle == 0:
            return image
        return image.rotate(360 - rotate_angle, expand=True)
    except Exception as e:
        print(f"[❌] Orientation detection failed: {e}")
        return image


def deskew_image(pil_img: Image.Image) -> Image.Image:
    img_cv = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return pil_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_cv = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    print(f"[🧭] Deskewed by {angle:.2f} degrees")
    return Image.fromarray(rotated_cv).convert("RGB")


def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)

    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    contrast = ImageEnhance.Contrast(gray).enhance(1.05)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.1)

    return sharpened


@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        oriented = correct_orientation(image)
        deskewed = deskew_image(oriented)
        enhanced = enhance_image(deskewed)

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

        oriented = correct_orientation(image)
        deskewed = deskew_image(oriented)
        enhanced = enhance_image(deskewed)

        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
