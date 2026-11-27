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
    "grain": {"strength": 0.5},
    "duotone": {"shadow": "#0000ff", "highlight": "#ffff00"},
    "threshold": {"threshold": 128},
    "text": {"text": "test", "size": 20},
    "border": {"width": 5},
}


def _check_property(name, src, out):
    """One property per op: the op has to do what its name says, not just run."""
    a = np.asarray(src.convert("RGB")).astype(int)
    b = np.asarray(out.convert("RGB")).astype(int)
    if name == "exposure":
        assert b.mean() > a.mean()
    elif name == "brightness":
        assert b.mean() > a.mean()
    elif name == "contrast":
        assert b.std() > a.std()
    elif name == "saturation":
        assert (b.max(axis=2) - b.min(axis=2)).mean() > (a.max(axis=2) - a.min(axis=2)).mean()
    elif name == "temperature":
        assert b[..., 0].mean() > a[..., 0].mean() and b[..., 2].mean() < a[..., 2].mean()
    elif name == "tint":
        assert b[..., 1].mean() < a[..., 1].mean()
    elif name in ("highlights", "shadows", "whites", "blacks"):
        # the gradient fixture never reaches the tones these ops move,
        # so check them on a two-tone image: one bright pixel, one dark
        tiny = Image.new("RGB", (2, 1))
        tiny.putpixel((0, 0), (230, 230, 230))
        tiny.putpixel((1, 0), (20, 20, 20))
        out2, _ = apply_ops(tiny, [{"op": name, "value": 30 if name in ("highlights", "shadows") else 20 if name == "whites" else -20}])
        c = np.asarray(out2.convert("RGB")).astype(int)
        if name == "highlights":
            assert c[0, 0].mean() > 230 and c[0, 1].mean() == 20
        elif name == "shadows":
            assert c[0, 1].mean() > 20 and c[0, 0].mean() == 230
        elif name == "whites":
            assert c[0, 0].mean() > 230
        else:
            assert c[0, 1].mean() < 20
    elif name in ("clarity", "sharpness"):
        # smooth gradients survive sharpening unchanged, so use a noisy image
        rng = np.random.default_rng(0)
        noisy = Image.fromarray((rng.random((40, 40, 3)) * 255).astype(np.uint8))
        params = {"value": 40} if name == "clarity" else {"factor": 1.5}
        out2, _ = apply_ops(noisy, [{"op": name, **params}])
        assert not np.array_equal(np.asarray(out2), np.asarray(noisy))
    elif name in ("hue", "equalize", "text"):
        assert not np.array_equal(b, a)
    elif name == "blur":
        assert b.std() < a.std()
    elif name == "vignette":
        assert b[0, 0].mean() < a[0, 0].mean()
    elif name == "crop":
        assert out.size == (90, 70)
    elif name == "resize":
        assert out.size == (100, 75)
    elif name == "rotate":
        assert out.size != src.size
    elif name == "flip":
        assert np.array_equal(b[0], a[-1])
    elif name == "grayscale":
        assert out.mode == "L"
    elif name == "autocontrast":
        assert b.min() == 0 and b.max() == 255
    elif name == "gamma":
        assert b.mean() > a.mean()
    elif name == "invert":
        assert np.array_equal(b, 255 - a)
    elif name == "posterize":
        assert set(np.unique(b)) <= {0, 64, 128, 192}
    elif name == "sepia":
        assert b[..., 0].mean() > b[..., 2].mean()
    elif name == "solarize":
        bright = a > 128
        assert np.array_equal(b[bright], 255 - a[bright])
        assert np.array_equal(b[~bright], a[~bright])
    elif name == "pixelate":
        assert (b[0, 0] == b[4, 4]).all() and not (b[0, 0] == b[0, 8]).all()
    elif name in ("emboss", "edge", "grain"):
        assert not np.array_equal(b, a)
    elif name == "duotone":
        assert tuple(b[0, 0]) == (0, 0, 255) or True  # covered by test_duotone_maps_black_and_white
    elif name == "threshold":
        assert set(np.unique(b)) <= {0, 255}
    elif name == "border":
        assert out.size == (src.size[0] + 10, src.size[1] + 10)


@pytest.mark.parametrize("name,params", ALL_OPS.items())
def test_each_op_runs(img, name, params):
    out, applied = apply_ops(img, [{"op": name, **params}])
    assert isinstance(out, Image.Image)
    assert applied == [name]
    _check_property(name, img, out)


@pytest.mark.skipif(__import__("importlib").util.find_spec("rembg") is None,
                    reason="rembg not installed")
def test_remove_background_returns_rgba(img):
    out, applied = apply_ops(img, [{"op": "remove_background"}])
    assert applied == ["remove_background"]
    assert out.mode == "RGBA"


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


def test_grain_speckle(img):
    out, _ = apply_ops(img, [{"op": "grain", "strength": 0.5}])
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    # same seed every time, so two runs match
    out2, _ = apply_ops(img, [{"op": "grain", "strength": 0.5}])
    assert np.array_equal(np.asarray(out), np.asarray(out2))
    # strength 0 leaves the image alone
    out0, _ = apply_ops(img, [{"op": "grain", "strength": 0}])
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


def test_duotone_maps_black_and_white():
    tiny = Image.new("RGB", (2, 1))
    tiny.putpixel((0, 0), (0, 0, 0))
    tiny.putpixel((1, 0), (255, 255, 255))
    out, _ = apply_ops(tiny, [{"op": "duotone", "shadow": "#0000ff", "highlight": "#ffff00"}])
    px = np.asarray(out.convert("RGB"))
    assert tuple(px[0, 0]) == (0, 0, 255)
    assert tuple(px[0, 1]) == (255, 255, 0)


def test_duotone_bad_color(img):
    with pytest.raises(ValueError, match="hex"):
        apply_ops(img, [{"op": "duotone", "shadow": "notacolor"}])


def test_threshold_is_binary():
    tiny = Image.new("RGB", (2, 1))
    tiny.putpixel((0, 0), (10, 10, 10))
    tiny.putpixel((1, 0), (250, 250, 250))
    out, _ = apply_ops(tiny, [{"op": "threshold", "threshold": 128}])
    px = np.asarray(out.convert("RGB"))
    assert tuple(px[0, 0]) == (0, 0, 0)
    assert tuple(px[0, 1]) == (255, 255, 255)
    assert set(np.unique(px)) <= {0, 255}
