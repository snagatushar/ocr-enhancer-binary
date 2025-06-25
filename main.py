from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

# Rotation disabled
def correct_orientation(image: Image.Image) -> Image.Image:
    print("[🚫] Auto-rotation skipped.")
    return image  # No rotation applied

# Deskew logic removed
def deskew_image(pil_img: Image.Image) -> Image.Image:
    print("[🚫] Deskew skipped.")
    return pil_img  # No deskewing

# Only enhancement logic remains
def enhance_image(image: Image.Image) -> Image.Image:
    print("[✨] Applying enhancement...")
    gray = ImageOps.grayscale(image)

    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    contrast = ImageEnhance.Contrast(gray).enhance(1.05)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.1)

    return sharpened


@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    """
    Returns the original image without alignment.
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Skip all corrections
        final_image = image

        img_bytes = BytesIO()
        final_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=aligned_image.png"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    """
    Returns enhanced image (no rotation or deskew).
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        enhanced = enhance_image(image)

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
    """
    Extracts text from enhanced image (no rotation or deskew).
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        enhanced = enhance_image(image)

        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
