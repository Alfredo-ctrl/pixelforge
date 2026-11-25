import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from pixelforge import apply_ops, list_ops, load_image


@pytest.fixture
def img():
    arr = np.zeros((150, 200, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, 200).astype(np.uint8)
    arr[..., 1] = 100
    arr[..., 2] = np.linspace(255, 0, 150)[:, None].astype(np.uint8)
    return Image.fromarray(arr)


ALL_OPS = {
    "exposure": {"ev": 0.5},
    "brightness": {"factor": 1.2},
    "contrast": {"factor": 1.2},
    "saturation": {"factor": 1.2},
    "temperature": {"value": 20},
    "tint": {"value": -10},
    "highlights": {"value": 30},
    "shadows": {"value": 30},
    "whites": {"value": 20},
    "blacks": {"value": -20},
    "clarity": {"value": 40},
    "sharpness": {"factor": 1.5},
    "blur": {"radius": 2},
    "vignette": {"strength": 30},
    "crop": {"left": 10, "top": 10, "right": 100, "bottom": 80},
    "resize": {"width": 100},
    "rotate": {"angle": 15},
    "flip": {"direction": "vertical"},
    "grayscale": {},
    "autocontrast": {},
    "gamma": {"value": 1.5},
    "hue": {"degrees": 30},
    "invert": {},
    "posterize": {"bits": 2},
    "equalize": {},
    "sepia": {"intensity": 1},
    "solarize": {"threshold": 128},
    "pixelate": {"size": 8},
    "emboss": {"strength": 1},
    "edge": {"strength": 1},
    "text": {"text": "test", "size": 20},
    "border": {"width": 5},
}


@pytest.mark.parametrize("name,params", ALL_OPS.items())
def test_each_op_runs(img, name, params):
    out, applied = apply_ops(img, [{"op": name, **params}])
    assert isinstance(out, Image.Image)
    assert applied == [name]


def test_resize_changes_size(img):
    out, _ = apply_ops(img, [{"op": "resize", "width": 100}])
    assert out.size == (100, 75)


def test_grayscale_mode(img):
    out, _ = apply_ops(img, [{"op": "grayscale"}])
    assert out.mode == "L"


def test_unknown_op_errors(img):
    with pytest.raises(ValueError, match="unknown op"):
        apply_ops(img, [{"op": "nope"}])


def test_out_of_range_errors(img):
    with pytest.raises(ValueError, match="out of range"):
        apply_ops(img, [{"op": "exposure", "ev": 99}])


def test_list_ops_schema():
    ops = list_ops()
    assert "exposure" in ops
    assert ops["exposure"]["params"]["ev"]["max"] == 3


def test_posterize_limits_colors(img):
    out, _ = apply_ops(img, [{"op": "posterize", "bits": 1}])
    assert out.size == img.size
    # 1 bit per channel leaves only 0 and 128 as possible values
    assert set(np.unique(np.asarray(out.convert("RGB")))) <= {0, 128}


def test_equalize_keeps_size_and_changes_pixels(img):
    out, _ = apply_ops(img, [{"op": "equalize"}])
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_sepia_warms_pixels(img):
    out, _ = apply_ops(img, [{"op": "sepia", "intensity": 1}])
    assert out.size == img.size
    arr = np.asarray(out.convert("RGB")).astype(int)
    # sepia pushes red above blue on average
    assert arr[..., 0].mean() > arr[..., 2].mean()
    # intensity 0 leaves the image alone
    out0, _ = apply_ops(img, [{"op": "sepia", "intensity": 0}])
    assert np.array_equal(np.asarray(out0), np.asarray(img))


def test_solarize_flips_bright_tones(img):
    out, _ = apply_ops(img, [{"op": "solarize", "threshold": 128}])
    assert out.size == img.size
    src = np.asarray(img.convert("RGB")).astype(int)
    got = np.asarray(out.convert("RGB")).astype(int)
    # bright tones flip to their negative, dark tones stay put
    bright = src > 128
    assert np.array_equal(got[bright], 255 - src[bright])
    assert np.array_equal(got[~bright], src[~bright])


def test_pixelate_blocks_pixels(img):
    out, _ = apply_ops(img, [{"op": "pixelate", "size": 8}])
    assert out.size == img.size
    arr = np.asarray(out.convert("RGB"))
    # every pixel inside an 8x8 block is identical
    assert (arr[0, 0] == arr[4, 4]).all()
    assert (arr[0, 0] == arr[7, 7]).all()
    # neighbour blocks differ on a gradient
    assert not (arr[0, 0] == arr[0, 8]).all()


def test_emboss_relief(img):
    out, _ = apply_ops(img, [{"op": "emboss", "strength": 1}])
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    # strength 0 leaves the image alone
    out0, _ = apply_ops(img, [{"op": "emboss", "strength": 0}])
    assert np.array_equal(np.asarray(out0), np.asarray(img))


def test_edge_outlines(img):
    out, _ = apply_ops(img, [{"op": "edge", "strength": 1}])
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    # flat areas go dark, only edges stay bright
    arr = np.asarray(out.convert("RGB")).astype(int)
    assert arr.mean() < np.asarray(img).mean()
    # strength 0 leaves the image alone
    out0, _ = apply_ops(img, [{"op": "edge", "strength": 0}])
    assert np.array_equal(np.asarray(out0), np.asarray(img))


def test_cli_edit_roundtrip(tmp_path, img):
    src = tmp_path / "in.jpg"
    dst = tmp_path / "out.jpg"
    img.save(src)
    r = subprocess.run(
        [sys.executable, "-m", "pixelforge.cli", "edit", "-i", str(src), "-o", str(dst),
         "--ops", json.dumps([{"op": "temperature", "value": 15}])],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["ops_applied"] == ["temperature"]
    assert load_image(str(dst)).size == img.size
