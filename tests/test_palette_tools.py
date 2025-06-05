# tests/test_palette_tools.py
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
import pytest
from pbn import palette_tools
import warnings
import pytest

# Suppress sklearn's ConvergenceWarning for single-color images
pytestmark = pytest.mark.filterwarnings("ignore:.*Number of distinct clusters.*:UserWarning")

def test_extract_palette_from_dummy_image(tmp_path):
    # Programmatically create the dummy image
    img = Image.new("RGB", (256, 256), color=(150, 120, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(50, 50), (150, 150)], fill=(200, 50, 50))
    draw.ellipse([(100, 100), (200, 200)], fill=(50, 200, 50))

    img_path = tmp_path / "dummy_input.png"
    img.save(img_path)

    # Extract palette
    max_colors = 3
    palette = palette_tools.extract_palette_from_image(str(img_path), max_colors=max_colors)

    # Check shape
    assert palette.shape == (max_colors, 3)
    assert palette.dtype == np.uint8

    # Check approximate match for each expected color
    expected_colors = [
        (150, 120, 200),  # background
        (200, 50, 50),    # red rectangle
        (50, 200, 50)     # green ellipse
    ]
    for expected_color in expected_colors:
        assert any(
            np.linalg.norm(np.array(expected_color) - extracted_color) < 15
            for extracted_color in palette
        ), f"Expected color {expected_color} not close to any extracted palette color."


def test_extract_palette_respects_max_colors():
    # In-memory 3-color image
    img = Image.new("RGB", (30, 10))
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, color in enumerate(colors):
        img.paste(color, (i * 10, 0, (i + 1) * 10, 10))

    # Save to BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Save to temp file for palette_tools
    with pytest.MonkeyPatch.context() as mp:
        tmpfile = "_test_max_colors.png"
        with open(tmpfile, "wb") as f:
            f.write(buf.read())

        requested_colors = 5
        palette = palette_tools.extract_palette_from_image(tmpfile, max_colors=requested_colors)

        assert palette.shape[0] == requested_colors
        assert palette.shape[1] == 3
        assert palette.dtype == np.uint8


def test_extract_palette_with_single_color_image():
    # Single-color in-memory image
    color = (123, 222, 64)
    img = Image.new("RGB", (10, 10), color=color)

    # Save to BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Save to temp file for palette_tools
    with pytest.MonkeyPatch.context() as mp:
        tmpfile = "_test_single_color.png"
        with open(tmpfile, "wb") as f:
            f.write(buf.read())

        palette = palette_tools.extract_palette_from_image(tmpfile, max_colors=4)

        assert palette.shape == (4, 3)
        for row in palette:
            assert np.linalg.norm(np.array(row) - np.array(color)) < 10


def test_extract_palette_invalid_path():
    with pytest.raises(FileNotFoundError):
        palette_tools.extract_palette_from_image("nonexistent_file.png", max_colors=3)
