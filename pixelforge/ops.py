"""Operations registry. Every op is a small function: image in, image out.

To add one: write the function, then register it in OPS with its params.
Params use a tiny schema so agents can discover them: type, default, min/max
or enum, and a one-line description.
"""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


def _split(img):
    # work on RGB, keep the alpha channel aside so it survives the edit
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        return Image.merge("RGB", (r, g, b)), a
    return img.convert("RGB"), None


def _join(rgb, alpha):
    if alpha is None:
        return rgb
    r, g, b = rgb.split()
    return Image.merge("RGBA", (r, g, b, alpha))


def _arr(img):
    return np.asarray(img).astype(np.float32)


def _from_arr(arr, alpha):
    return _join(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), alpha)


def _lum(arr):
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


# --- light ---


def op_exposure(img, ev=0.0):
    rgb, a = _split(img)
    return _from_arr(_arr(rgb) * (2.0 ** ev), a)


def op_brightness(img, factor=1.0):
    return ImageEnhance.Brightness(img).enhance(factor)


def op_contrast(img, factor=1.0):
    return ImageEnhance.Contrast(img).enhance(factor)


def op_saturation(img, factor=1.0):
    return ImageEnhance.Color(img).enhance(factor)


def op_temperature(img, value=0.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    arr[..., 0] += value * 0.6
    arr[..., 2] -= value * 0.6
    return _from_arr(arr, a)


def op_tint(img, value=0.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    arr[..., 1] += value * 0.6
    return _from_arr(arr, a)


def op_highlights(img, value=0.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    lum = _lum(arr) / 255.0
    mask = np.clip((lum - 0.5) * 2, 0, 1) ** 2
    arr += (value * 0.5) * mask[..., None]
    return _from_arr(arr, a)


def op_shadows(img, value=0.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    lum = _lum(arr) / 255.0
    mask = np.clip((0.5 - lum) * 2, 0, 1) ** 2
    arr += (value * 0.5) * mask[..., None]
    return _from_arr(arr, a)


def op_whites(img, value=0.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    lum = _lum(arr) / 255.0
    mask = np.clip((lum - 0.75) * 4, 0, 1) ** 2
    arr += (value * 0.4) * mask[..., None]
    return _from_arr(arr, a)


def op_blacks(img, value=0.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    lum = _lum(arr) / 255.0
    mask = np.clip((0.25 - lum) * 4, 0, 1) ** 2
    arr += (value * 0.4) * mask[..., None]
    return _from_arr(arr, a)


def op_clarity(img, value=0.0):
    if value == 0:
        return img
    sharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(value), threshold=2))
    return Image.blend(img, sharp, 0.6)


def op_sharpness(img, factor=1.0):
    return ImageEnhance.Sharpness(img).enhance(factor)


def op_blur(img, radius=0.0):
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius))


def op_vignette(img, strength=0.0):
    if strength <= 0:
        return img
    rgb, a = _split(img)
    w, h = rgb.size
    y, x = np.mgrid[0:h, 0:w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1 - (strength / 100.0) * np.clip(d - 0.5, 0, None) ** 2 * 2, 0, 1)
    return _from_arr(_arr(rgb) * mask[..., None], a)


def op_gamma(img, value=1.0):
    if value == 1.0:
        return img
    rgb, a = _split(img)
    arr = _arr(rgb) / 255.0
    return _from_arr(np.power(arr, 1.0 / value) * 255.0, a)


def op_hue(img, degrees=0.0):
    if degrees == 0:
        return img
    rgb, a = _split(img)
    hsv = np.asarray(rgb.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(degrees / 360 * 255)) % 255
    out = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    return _join(out, a)


def op_grayscale(img):
    return ImageOps.grayscale(img)


def op_autocontrast(img, cutoff=0):
    rgb, a = _split(img)
    return _join(ImageOps.autocontrast(rgb, cutoff=cutoff), a)


def op_invert(img):
    rgb, a = _split(img)
    return _join(ImageOps.invert(rgb), a)


def op_posterize(img, bits=4):
    rgb, a = _split(img)
    return _join(ImageOps.posterize(rgb, bits), a)


def op_equalize(img):
    rgb, a = _split(img)
    return _join(ImageOps.equalize(rgb), a)


def op_sepia(img, intensity=1.0):
    rgb, a = _split(img)
    arr = _arr(rgb)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    sr = r * 0.393 + g * 0.769 + b * 0.189
    sg = r * 0.349 + g * 0.686 + b * 0.168
    sb = r * 0.272 + g * 0.534 + b * 0.131
    sepia = np.stack([sr, sg, sb], axis=-1)
    return _from_arr(arr + (sepia - arr) * intensity, a)


def op_solarize(img, threshold=128):
    # flip the bright tones, keep the dark ones — classic darkroom look
    rgb, a = _split(img)
    arr = _arr(rgb)
    out = np.where(arr > threshold, 255 - arr, arr)
    return _from_arr(out, a)


def op_pixelate(img, size=8):
    # blocky mosaic: shrink it down, then blow it back up chunky
    rgb, a = _split(img)
    w, h = rgb.size
    small = rgb.resize((max(1, w // size), max(1, h // size)), Image.NEAREST)
    big = small.resize((w, h), Image.NEAREST)
    return _from_arr(np.asarray(big).astype(np.float32), a)


def op_emboss(img, strength=1.0):
    # raised relief look, like stamped metal; 0 keeps the original
    if strength <= 0:
        return img
    rgb, a = _split(img)
    emb = rgb.filter(ImageFilter.EMBOSS)
    if strength >= 1.0:
        return _join(emb, a)
    arr = _arr(rgb)
    return _from_arr(arr + (_arr(emb) - arr) * strength, a)


def op_edge(img, strength=1.0):
    # outlines only, like a pencil sketch; 0 keeps the original
    if strength <= 0:
        return img
    rgb, a = _split(img)
    edged = rgb.filter(ImageFilter.FIND_EDGES)
    if strength >= 1.0:
        return _join(edged, a)
    arr = _arr(rgb)
    return _from_arr(arr + (_arr(edged) - arr) * strength, a)


def op_grain(img, strength=0.2):
    # film grain: random speckle like old analog photos; 0 keeps the original
    if strength <= 0:
        return img
    rgb, a = _split(img)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, strength * 40, _arr(rgb).shape[:2])[..., None]
    return _from_arr(_arr(rgb) + noise, a)


# --- geometry ---


def _hex_to_rgb(hexstr):
    # "#ff8800" -> (255, 136, 0). anything else is a bad color
    if not isinstance(hexstr, str):
        raise ValueError(f"color has to be hex text like '#ff8800', got {hexstr!r}")
    h = hexstr.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"color has to be 6 hex digits like '#ff8800', got {hexstr!r}")
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"color has to be 6 hex digits like '#ff8800', got {hexstr!r}")


def op_duotone(img, shadow="#000000", highlight="#ffffff"):
    # two-tone print: dark parts take the shadow color, bright parts the highlight
    rgb, a = _split(img)
    arr = _arr(rgb)
    lum = (_lum(arr) / 255.0)[..., None]
    lo = np.array(_hex_to_rgb(shadow), dtype=np.float32)
    hi = np.array(_hex_to_rgb(highlight), dtype=np.float32)
    out = lo * (1 - lum) + hi * lum
    return _from_arr(out, a)


def op_crop(img, left=0, top=0, right=0, bottom=0):
    w, h = img.size
    r = w if not right else right
    b = h if not bottom else bottom
    if not (0 <= left < r <= w and 0 <= top < b <= h):
        raise ValueError(f"crop box {(left, top, r, b)} outside image {w}x{h}")
    return img.crop((left, top, r, b))


def op_resize(img, width=0, height=0):
    w, h = img.size
    if width == 0 and height == 0:
        raise ValueError("resize needs width or height")
    if width == 0:
        width = round(w * height / h)
    if height == 0:
        height = round(h * width / w)
    return img.resize((width, height), Image.LANCZOS)


def op_rotate(img, angle=0.0, expand=True):
    return img.rotate(angle, expand=expand, resample=Image.BICUBIC)


def op_flip(img, direction="horizontal"):
    if direction == "horizontal":
        return ImageOps.mirror(img)
    if direction == "vertical":
        return ImageOps.flip(img)
    raise ValueError("flip direction must be horizontal or vertical")


def op_border(img, width=10, color="#000000"):
    return ImageOps.expand(img, border=int(width), fill=color)


def op_text(img, text="", x=10, y=10, size=36, color="#ffffff", font=""):
    if not text:
        raise ValueError("text needs a string")
    out = img.copy()
    draw = ImageDraw.Draw(out)
    fnt = ImageFont.truetype(font, size) if font else ImageFont.load_default(size)
    draw.text((x, y), text, fill=color, font=fnt)
    return out


def op_remove_background(img):
    try:
        from rembg import remove
    except ImportError:
        raise RuntimeError("remove_background needs the bg extra: pip install pixelforge[bg]")
    return remove(img)


# --- registry ---


def _p(type_, default, desc, **kw):
    d = {"type": type_, "default": default, "desc": desc}
    d.update(kw)
    return d


OPS = {
    "exposure": {
        "fn": op_exposure,
        "desc": "Exposure in stops. +1 doubles brightness.",
        "params": {"ev": _p("number", 0.0, "Exposure compensation in EV", min=-3, max=3)},
    },
    "brightness": {
        "fn": op_brightness,
        "desc": "Overall brightness multiplier. 1 keeps it as is.",
        "params": {"factor": _p("number", 1.0, "Brightness multiplier", min=0, max=3)},
    },
    "contrast": {
        "fn": op_contrast,
        "desc": "Contrast multiplier. 1 keeps it as is.",
        "params": {"factor": _p("number", 1.0, "Contrast multiplier", min=0, max=3)},
    },
    "saturation": {
        "fn": op_saturation,
        "desc": "Color intensity. 0 is black and white, 1 keeps it as is.",
        "params": {"factor": _p("number", 1.0, "Saturation multiplier", min=0, max=3)},
    },
    "temperature": {
        "fn": op_temperature,
        "desc": "White balance. Positive is warmer, negative is cooler.",
        "params": {"value": _p("number", 0.0, "Warm (+) / cool (-) shift", min=-100, max=100)},
    },
    "tint": {
        "fn": op_tint,
        "desc": "Green/magenta shift. Positive is greener.",
        "params": {"value": _p("number", 0.0, "Green (+) / magenta (-) shift", min=-100, max=100)},
    },
    "highlights": {
        "fn": op_highlights,
        "desc": "Lift or recover the brightest parts only.",
        "params": {"value": _p("number", 0.0, "Highlight adjustment", min=-100, max=100)},
    },
    "shadows": {
        "fn": op_shadows,
        "desc": "Lift or crush the darkest parts only.",
        "params": {"value": _p("number", 0.0, "Shadow adjustment", min=-100, max=100)},
    },
    "whites": {
        "fn": op_whites,
        "desc": "Adjust the extreme bright end of the histogram.",
        "params": {"value": _p("number", 0.0, "White point adjustment", min=-100, max=100)},
    },
    "blacks": {
        "fn": op_blacks,
        "desc": "Adjust the extreme dark end of the histogram.",
        "params": {"value": _p("number", 0.0, "Black point adjustment", min=-100, max=100)},
    },
    "clarity": {
        "fn": op_clarity,
        "desc": "Local contrast. Adds punch to midtone detail.",
        "params": {"value": _p("number", 0.0, "Clarity amount", min=0, max=100)},
    },
    "sharpness": {
        "fn": op_sharpness,
        "desc": "Edge sharpness multiplier. 1 keeps it as is.",
        "params": {"factor": _p("number", 1.0, "Sharpness multiplier", min=0, max=3)},
    },
    "blur": {
        "fn": op_blur,
        "desc": "Gaussian blur over the whole frame.",
        "params": {"radius": _p("number", 0.0, "Blur radius in pixels", min=0, max=20)},
    },
    "vignette": {
        "fn": op_vignette,
        "desc": "Darken the corners, keep the center.",
        "params": {"strength": _p("number", 0.0, "Vignette strength", min=0, max=100)},
    },
    "gamma": {
        "fn": op_gamma,
        "desc": "Gamma correction. Below 1 brightens midtones, above 1 darkens them.",
        "params": {"value": _p("number", 1.0, "Gamma value", min=0.1, max=3)},
    },
    "hue": {
        "fn": op_hue,
        "desc": "Rotate hues around the color wheel.",
        "params": {"degrees": _p("number", 0.0, "Hue rotation in degrees", min=-180, max=180)},
    },
    "grayscale": {
        "fn": op_grayscale,
        "desc": "Convert to black and white.",
        "params": {},
    },
    "autocontrast": {
        "fn": op_autocontrast,
        "desc": "Stretch the histogram so the darkest pixel is black and the brightest is white.",
        "params": {"cutoff": _p("number", 0, "Percent to cut from both ends", min=0, max=10)},
    },
    "invert": {
        "fn": op_invert,
        "desc": "Negative image.",
        "params": {},
    },
    "posterize": {
        "fn": op_posterize,
        "desc": "Cut the colors down to a few flat levels. Low bits means a poster look.",
        "params": {"bits": _p("integer", 4, "Bits per channel, 1 to 8. Lower means fewer colors.", min=1, max=8)},
    },
    "equalize": {
        "fn": op_equalize,
        "desc": "Spread the histogram out so the tones use the full range.",
        "params": {},
    },
    "sepia": {
        "fn": op_sepia,
        "desc": "Warm old-photo tone. 0 keeps the original, 1 is full sepia.",
        "params": {"intensity": _p("number", 1.0, "Sepia strength", min=0, max=1)},
    },
    "solarize": {
        "fn": op_solarize,
        "desc": "Flip tones above the threshold to their negative, keep the rest. Darkroom solarization look.",
        "params": {"threshold": _p("integer", 128, "Tones above this get inverted", min=0, max=255)},
    },
    "pixelate": {
        "fn": op_pixelate,
        "desc": "Blocky mosaic, big chunky pixels. Bigger size = chunkier.",
        "params": {"size": _p("integer", 8, "Pixel block size in px", min=2, max=64)},
    },
    "emboss": {
        "fn": op_emboss,
        "desc": "Raised relief look, like stamped metal. 0 keeps the original.",
        "params": {"strength": _p("number", 1.0, "Emboss strength", min=0, max=1)},
    },
    "edge": {
        "fn": op_edge,
        "desc": "Edge detection, outlines only like a pencil sketch. 0 keeps the original.",
        "params": {"strength": _p("number", 1.0, "Edge strength", min=0, max=1)},
    },
    "grain": {
        "fn": op_grain,
        "desc": "Film grain, random speckle like old analog photos. 0 keeps the original.",
        "params": {"strength": _p("number", 0.2, "Grain strength", min=0, max=1)},
    },
    "duotone": {
        "fn": op_duotone,
        "desc": "Two-tone print look: shadows take one color, highlights take another.",
        "params": {
            "shadow": _p("string", "#000000", "Color for the dark parts, as hex"),
            "highlight": _p("string", "#ffffff", "Color for the bright parts, as hex"),
        },
    },
    "crop": {
        "fn": op_crop,
        "desc": "Crop to a pixel box. Right/bottom default to the image edge.",
        "params": {
            "left": _p("integer", 0, "Left edge in pixels", min=0),
            "top": _p("integer", 0, "Top edge in pixels", min=0),
            "right": _p("integer", 0, "Right edge in pixels, 0 means image edge", min=0),
            "bottom": _p("integer", 0, "Bottom edge in pixels, 0 means image edge", min=0),
        },
    },
    "resize": {
        "fn": op_resize,
        "desc": "Resize. Set one side to 0 to keep the aspect ratio.",
        "params": {
            "width": _p("integer", 0, "Target width in pixels, 0 keeps aspect", min=0),
            "height": _p("integer", 0, "Target height in pixels, 0 keeps aspect", min=0),
        },
    },
    "rotate": {
        "fn": op_rotate,
        "desc": "Rotate counter-clockwise in degrees.",
        "params": {
            "angle": _p("number", 0.0, "Rotation angle in degrees", min=-360, max=360),
            "expand": _p("boolean", True, "Grow the canvas so nothing is cut off"),
        },
    },
    "flip": {
        "fn": op_flip,
        "desc": "Mirror the image.",
        "params": {"direction": _p("string", "horizontal", "Flip direction", enum=["horizontal", "vertical"])},
    },
    "border": {
        "fn": op_border,
        "desc": "Add a solid frame around the image.",
        "params": {
            "width": _p("integer", 10, "Border width in pixels", min=0),
            "color": _p("string", "#000000", "Border color as hex"),
        },
    },
    "text": {
        "fn": op_text,
        "desc": "Draw text on the image.",
        "params": {
            "text": _p("string", "", "Text to draw"),
            "x": _p("integer", 10, "X position in pixels", min=0),
            "y": _p("integer", 10, "Y position in pixels", min=0),
            "size": _p("integer", 36, "Font size in pixels", min=1),
            "color": _p("string", "#ffffff", "Text color as hex"),
            "font": _p("string", "", "Path to a .ttf file, empty means built-in font"),
        },
    },
    "remove_background": {
        "fn": op_remove_background,
        "desc": "Cut out the subject, transparent background. Needs pixelforge[bg].",
        "params": {},
    },
}


def list_ops():
    return {name: {"desc": o["desc"], "params": o["params"]} for name, o in OPS.items()}


def apply_ops(img, ops):
    if isinstance(ops, str):
        ops = json.loads(ops)
    out = img
    applied = []
    for step in ops:
        name = step.get("op")
        if name not in OPS:
            raise ValueError(f"unknown op '{name}'. Run `pixelforge ops` for the list.")
        params = {k: v for k, v in step.items() if k != "op"}
        spec = OPS[name]
        for key, val in params.items():
            if key not in spec["params"]:
                raise ValueError(f"op '{name}' has no param '{key}'. Valid: {sorted(spec['params'])}")
            p = spec["params"][key]
            if "min" in p and val < p["min"]:
                raise ValueError(f"op '{name}' param '{key}' out of range [{p['min']}, {p['max']}]: {val}")
            if "max" in p and val > p["max"]:
                raise ValueError(f"op '{name}' param '{key}' out of range [{p['min']}, {p['max']}]: {val}")
            if "enum" in p and val not in p["enum"]:
                raise ValueError(f"op '{name}' param '{key}' must be one of {p['enum']}: {val}")
        out = spec["fn"](out, **params)
        applied.append(name)
    return out, applied
