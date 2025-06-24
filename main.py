from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
from io import BytesIO

# Point to tesseract binary inside Docker
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()


def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    resized = gray.resize((gray.width * 2, gray.height * 2), Image.BICUBIC)
    filtered = resized.filter(ImageFilter.MedianFilter(size=3))
    contrast = ImageEnhance.Contrast(filtered).enhance(1.2)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.2)
    return sharpened


@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        enhanced = enhance_image(image)

        img_bytes = BytesIO()
        enhanced.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        enhanced = enhance_image(image)

        text = pytesseract.image_to_string(enhanced, config="--psm 6")

        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
