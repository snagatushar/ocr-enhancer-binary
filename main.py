from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()


def smart_deskew(pil_img: Image.Image, angle_threshold: float = 2.0) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        print("[ℹ️] No foreground pixels found, skipping deskew.")
        return pil_img

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) <= angle_threshold:
        print(f"[✅] Angle {angle:.2f}° is within threshold ({angle_threshold}°), skipping deskew.")
        return pil_img

    print(f"[🧭] Deskewing... angle detected: {angle:.2f}°")

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


@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    """
    Aligns image only if skew angle is more than 2 degrees.
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        aligned = smart_deskew(image, angle_threshold=2.0)

        img_bytes = BytesIO()
        aligned.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=aligned_image.png"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
