from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import numpy as np
import cv2
import io

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()


def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


def crop_image(pil_img: Image.Image) -> Image.Image:
    # You can change this box (left, upper, right, lower)
    width, height = pil_img.size
    box = (0, 0, width, height)  # No-op by default
    return pil_img.crop(box)


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

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(np.array(pil_img), M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(rotated)


def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)

    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    contrast = ImageEnhance.Contrast(gray).enhance(1.05)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.1)

    return sharpened


def run_ocr(image: Image.Image) -> str:
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(image, config=config)


@app.post("/ocr")
async def full_pipeline(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = load_image_from_bytes(image_data)

        cropped = crop_image(image)
        aligned = deskew_image(cropped)
        enhanced = enhance_image(aligned)
        text = run_ocr(enhanced)

        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/processed-image")
async def processed_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = load_image_from_bytes(image_data)

        cropped = crop_image(image)
        aligned = deskew_image(cropped)
        enhanced = enhance_image(aligned)

        buffer = io.BytesIO()
        enhanced.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
