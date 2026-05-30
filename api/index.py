import io
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from typing import Optional

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import pillow_avif
except ImportError:
    pass

app = FastAPI()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

SUPPORTED_FORMATS = {
    "jpeg": ("JPEG", "image/jpeg", ".jpg"),
    "jpg":  ("JPEG", "image/jpeg", ".jpg"),
    "png":  ("PNG",  "image/png",  ".png"),
    "webp": ("WEBP", "image/webp", ".webp"),
    "avif": ("AVIF", "image/avif", ".avif"),
    "gif":  ("GIF",  "image/gif",  ".gif"),
    "bmp":  ("BMP",  "image/bmp",  ".bmp"),
    "tiff": ("TIFF", "image/tiff", ".tiff"),
}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


def composite_on_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img)
        return bg
    return img


def apply_resize(img: Image.Image, width: Optional[int]) -> Image.Image:
    if not width or width <= 0 or width >= img.width:
        return img
    ratio = width / img.width
    new_h = max(1, round(img.height * ratio))
    return img.resize((width, new_h), Image.LANCZOS)


def apply_filters(img: Image.Image, filters: list[str]) -> Image.Image:
    for f in filters:
        if f == "grayscale":
            mode = img.mode
            img = img.convert("L")
            if mode == "RGBA":
                img = img.convert("RGBA")
        elif f == "sharpen":
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        elif f == "brighten":
            img = ImageEnhance.Brightness(img).enhance(1.35)
    return img


def apply_watermark(img: Image.Image, text: str) -> Image.Image:
    if not text.strip():
        return img

    # Work on RGBA copy so watermark blends correctly
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(12, min(img.width, img.height) // 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    margin = max(10, font_size // 2)
    x = img.width  - tw - margin
    y = img.height - th - margin

    # Shadow
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 100))
    # Text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 160))

    composited = Image.alpha_composite(base, overlay)

    if img.mode != "RGBA":
        composited = composited.convert(img.mode if img.mode != "P" else "RGB")
    return composited


def process_image(
    data: bytes,
    target_format: str,
    quality: int,
    resize_width: Optional[int],
    watermark: str,
    filters: list[str],
) -> tuple[bytes, str, str]:
    fmt_key = target_format.lower()
    if fmt_key not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {target_format}")

    pil_format, mime_type, ext = SUPPORTED_FORMATS[fmt_key]

    img = Image.open(io.BytesIO(data))
    is_animated = hasattr(img, "n_frames") and img.n_frames > 1

    # --- Edits (resize → filter → watermark) ---
    img = apply_resize(img, resize_width)
    img = apply_filters(img, filters)
    if watermark.strip():
        img = apply_watermark(img, watermark.strip())

    # --- Mode normalisation for target format ---
    if pil_format == "JPEG":
        img = composite_on_white(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    elif pil_format == "PNG":
        if img.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            img = img.convert("RGBA")
    elif pil_format in ("WEBP", "AVIF"):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
    else:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

    buf = io.BytesIO()

    if pil_format == "GIF" and is_animated:
        frames = []
        try:
            for i in range(img.n_frames):
                img.seek(i)
                frames.append(img.copy())
        except EOFError:
            pass
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0)
        return buf.getvalue(), mime_type, ext

    save_kwargs: dict = {}
    if pil_format == "JPEG":
        save_kwargs = {"quality": quality, "optimize": True}
    elif pil_format == "WEBP":
        save_kwargs = {"quality": quality, "method": 6}
    elif pil_format == "PNG":
        save_kwargs = {"optimize": True}
    elif pil_format == "AVIF":
        save_kwargs = {"quality": quality}

    img.save(buf, format=pil_format, **save_kwargs)
    return buf.getvalue(), mime_type, ext


def streaming_response(output: bytes, mime: str, filename: str, orig_size: int) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(output),
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(output)),
            "X-Original-Size": str(orig_size),
            "X-Output-Size": str(len(output)),
        },
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    format: str = Form(...),
    quality: int = Form(85),
    resize_width: int = Form(0),
    watermark: str = Form(""),
    filters: str = Form(""),          # comma-separated, e.g. "grayscale,sharpen"
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기가 10MB를 초과합니다.")
    if not 1 <= quality <= 100:
        quality = 85

    filter_list = [f.strip() for f in filters.split(",") if f.strip()]
    output, mime, ext = process_image(
        contents, format, quality,
        resize_width if resize_width > 0 else None,
        watermark, filter_list,
    )

    stem = os.path.splitext(file.filename or "image")[0]
    return streaming_response(output, mime, f"{stem}_slimpic{ext}", len(contents))


@app.post("/api/optimize")
async def optimize(
    file: UploadFile = File(...),
    resize_width: int = Form(0),
    watermark: str = Form(""),
    filters: str = Form(""),
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기가 10MB를 초과합니다.")

    filter_list = [f.strip() for f in filters.split(",") if f.strip()]
    output, mime, ext = process_image(
        contents, "webp", 82,
        resize_width if resize_width > 0 else None,
        watermark, filter_list,
    )

    stem = os.path.splitext(file.filename or "image")[0]
    return streaming_response(output, mime, f"{stem}_optimized.webp", len(contents))
