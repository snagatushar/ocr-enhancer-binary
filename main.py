from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

# Optional crop box for specific area
def crop_image(pil_img: Image.Image, box=(0, 0, 2000, 2000)) -> Image.Image:
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
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(rotated)

def enhance_image(pil_img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(pil_img)
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)
    contrast = ImageEnhance.Contrast(gray).enhance(1.05)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.1)
    return sharpened.rotate(0)  # No extra rotation

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        original = Image.open(BytesIO(image_data)).convert("RGB")

        # Step-by-step pipeline
        cropped = crop_image(original)
        aligned = deskew_image(cropped)
        enhanced = enhance_image(aligned)
        text = pytesseract.image_to_string(enhanced, config="--psm 6")

        # Return image + text
        img_bytes = BytesIO()
        enhanced.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes, media_type="image/png", headers={
                "Content-Disposition": "inline; filename=processed.png",
                "X-OCR-Text": text.strip()
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
