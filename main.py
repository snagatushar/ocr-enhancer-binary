from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import pytesseract
import numpy as np
import cv2
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

def deskew_image(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))

    # Threshold to find text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get coordinates of the non-white pixels (text)
    coords = np.column_stack(np.where(thresh < 255))
    if coords.shape[0] == 0:
        return pil_img  # nothing to align

    angle = cv2.minAreaRect(coords)[-1]

    # Correct angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    print(f"[🧭] Detected rotation angle: {angle:.2f}°")

    # Rotation matrix
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust matrix to avoid cropping
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(np.array(pil_img), M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

    return Image.fromarray(rotated)


def enhance_image(image: Image.Image) -> Image.Image:
    img_cv = np.array(image.convert("L"))

    # Resize if small
    if img_cv.shape[1] < 1200:
        img_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold (preserve faint lines)
    processed = cv2.adaptiveThreshold(
        img_cv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Debug: Save intermediate image (optional)
    Image.fromarray(processed).save("debug_ocr_image.png")

    return Image.fromarray(processed)


@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        deskewed = deskew_image(image)
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

        deskewed = deskew_image(image)
        enhanced = enhance_image(deskewed)

        config = "--oem 1 --psm 3"
        text = pytesseract.image_to_string(enhanced, config=config)

        return {"text": text.strip()}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
