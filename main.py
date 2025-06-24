from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
from io import BytesIO

# Required when running inside Docker on Render
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        # Step 1: Load image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Step 2: Grayscale
        gray = ImageOps.grayscale(image)

        # Step 3: Resize
        upscaled = gray.resize((gray.width * 2, gray.height * 2), Image.BICUBIC)

        # Step 4: Slight filtering (not overdone)
        filtered = upscaled.filter(ImageFilter.MedianFilter(size=3))

        # Step 5: Moderate contrast + sharpness (don't overdo it)
        contrast = ImageEnhance.Contrast(filtered).enhance(1.5)
        sharpened = ImageEnhance.Sharpness(contrast).enhance(1.5)

        # Step 6: OCR
        text = pytesseract.image_to_string(sharpened, config='--psm 6')
        print("🔍 OCR Text:\n", text)

        # Step 7: Convert to binary
        img_bytes = BytesIO()
        sharpened.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Step 8: Return binary image
        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
