from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import re

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()


def correct_orientation(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image)
        rotate_angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        print(f"[🔁] Detected orientation: {rotate_angle}°")

        if rotate_angle == 0:
            return image
        return image.rotate(360 - rotate_angle, expand=True)
    except Exception as e:
        print(f"[❌] Orientation detection failed: {e}")
        return image


def deskew_image(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh < 255))
    if coords.shape[0] == 0:
        return pil_img

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    print(f"[🧭] Deskew angle: {angle:.2f}°")

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(np.array(pil_img), M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(rotated)


def enhance_image(image: Image.Image) -> Image.Image:
    img_cv = np.array(image.convert("L"))

    if img_cv.shape[1] < 1200:
        img_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Denoise background while preserving edges
    img_denoised = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive threshold (more robust)
    processed = cv2.adaptiveThreshold(
        img_denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        5
    )

    # Debug save
    Image.fromarray(processed).save("debug_ocr_image.png")

    return Image.fromarray(processed)


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

        config = "--oem 1 --psm 3"
        text = pytesseract.image_to_string(enhanced, config=config)

        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
