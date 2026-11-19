"""Make a sample photo and give it a cinematic grade.

Run: python examples/cinematic.py
"""

import os

import numpy as np
from PIL import Image, ImageDraw

from pixelforge import apply_ops, save_image

OUT = "/tmp/pixelforge-example"
os.makedirs(OUT, exist_ok=True)

# fake sunset photo: gradient sky, sun, dark ground
w, h = 800, 500
arr = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    t = y / h
    arr[y, :, 0] = int(30 + 200 * t)
    arr[y, :, 1] = int(20 + 120 * t)
    arr[y, :, 2] = int(90 - 60 * t)
img = Image.fromarray(arr)
draw = ImageDraw.Draw(img)
draw.ellipse([330, 120, 470, 260], fill=(255, 220, 130))
draw.rectangle([0, 380, w, h], fill=(20, 25, 30))

save_image(img, f"{OUT}/before.jpg")

graded, applied = apply_ops(img, [
    {"op": "temperature", "value": 15},
    {"op": "contrast", "factor": 1.15},
    {"op": "saturation", "factor": 1.1},
    {"op": "vignette", "strength": 30},
])
save_image(graded, f"{OUT}/after.jpg")

print(f"before: {OUT}/before.jpg")
print(f"after:  {OUT}/after.jpg")
print(f"ops: {', '.join(applied)}")
