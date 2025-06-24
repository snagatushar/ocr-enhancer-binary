from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
from io import BytesIO

app = FastAPI()

@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        # Step 1: Read and convert to RGB
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Step 2–6: Enhancement
        gray = ImageOps.grayscale(image)
        upscaled = gray.resize((gray.width * 2, gray.height * 2), Image.BICUBIC)
        denoised = upscaled.filter(ImageFilter.MedianFilter(size=3))
        contrast = ImageEnhance.Contrast(denoised).enhance(2.0)
        sharpened = ImageEnhance.Sharpness(contrast).enhance(2.0)

        # Step 7: OCR (optional – you can print or log it)
        text = pytesseract.image_to_string(sharpened, config='--psm 6')
        print("[Extracted Text]\n", text)

        # Step 8: Return as binary image
        img_bytes = BytesIO()
        sharpened.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
