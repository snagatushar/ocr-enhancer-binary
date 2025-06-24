from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import base64

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

def deskew_image(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return pil_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # OpenCV returns angle in range [-90, 0)
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle

    # Only deskew if angle is between -15 and 15 degrees
    if abs(angle) < 0.5 or abs(angle) > 15:
        return pil_img

    # Negative angle means rotate clockwise, positive means counter-clockwise
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h),
                             flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(rotated).convert("RGB")

def enhance_image(image: Image.Image) -> Image.Image:
    # 1. Grayscale
    gray = ImageOps.grayscale(image)

    # 2. Resize if small
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    # 3. Denoise (median filter)
    denoised = gray.filter(ImageFilter.MedianFilter(size=3))

    # 4. Contrast and sharpness
    contrast = ImageEnhance.Contrast(denoised).enhance(1.3)  # Increased
    sharp = ImageEnhance.Sharpness(contrast).enhance(2.0)    # Increased

    # 5. Adaptive threshold (binarization)
    np_img = np.array(sharp)
    # Use Otsu's thresholding
    _, binarized = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final = Image.fromarray(binarized).convert("RGB")

    return final

@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image(image)

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
        # --- Step 1: Load binary image from request ---
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # --- Step 2: Grayscale ---
        gray = ImageOps.grayscale(image)

        # --- Step 3: Resize (upscale) ---
        upscaled = gray.resize((gray.width * 2, gray.height * 2), Image.BICUBIC)

        # --- Step 4: Median Filter (Denoise) ---
        denoised = upscaled.filter(ImageFilter.MedianFilter(size=3))

        # --- Step 5: Enhance Contrast ---
        contrast = ImageEnhance.Contrast(denoised).enhance(2.0)

        # --- Step 6: Enhance Sharpness ---
        sharpened = ImageEnhance.Sharpness(contrast).enhance(2.0)

        # --- Step 7: OCR using Tesseract ---
        text = pytesseract.image_to_string(sharpened, config='--psm 6')

        # --- Step 8: Convert final image to binary (base64) ---
        img_bytes = BytesIO()
        sharpened.save(img_bytes, format='PNG')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

        return {
            "text": text,
            "enhanced_image_base64": img_base64
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image(image)
        enhanced = enhance_image(aligned)

        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return {"text": text.strip()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) 