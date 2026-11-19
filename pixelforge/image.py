"""Load and save images. Keeps it boring on purpose."""

import os

from PIL import Image, ImageOps

FORMAT_FOR_SUFFIX = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tiff": "TIFF",
}


def load_image(path):
    img = Image.open(path)
    fmt = img.format
    img = ImageOps.exif_transpose(img)
    img.format = fmt
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    return img


def save_image(img, path, quality=90):
    suffix = os.path.splitext(path)[1].lower()
    fmt = FORMAT_FOR_SUFFIX.get(suffix)
    if fmt is None:
        raise ValueError(f"unsupported output format '{suffix}'. Use .jpg, .png, .webp, .bmp or .tiff")
    out = img
    if fmt == "JPEG" and out.mode in ("RGBA", "LA", "P"):
        out = out.convert("RGB")
    kwargs = {}
    if fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = quality
    if fmt == "PNG":
        kwargs["optimize"] = True
    out.save(path, format=fmt, **kwargs)
    return fmt
