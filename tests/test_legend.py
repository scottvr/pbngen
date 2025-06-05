# tests/test_legend.py
from pathlib import Path
from PIL import Image
import numpy as np
from pbn import legend


def test_create_legend_image_returns_image(tmp_path):
    # Create a dummy palette: 3 RGB colors
    palette = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255)     # blue
    ]

    # Generate legend image
    legend_image = legend.create_legend_image(palette, font_size=12, swatch_size=20, padding=5)

    # Validate the result is a PIL Image
    assert isinstance(legend_image, Image.Image)

    # Validate image size matches calculated expected size
    num_colors = len(palette)
    expected_width = (20 * num_colors) + (5 * (num_colors + 1))
    expected_height = 20 + (2 * 5)
    assert legend_image.size == (expected_width, expected_height)

    # Save for visual debugging (optional)
    outpath = tmp_path / "legend_test_output.png"
    legend_image.save(outpath)

    # Validate the saved file exists
    assert outpath.exists()


def test_create_legend_image_with_empty_palette():
    # Empty palette should return None
    result = legend.create_legend_image([])
    assert result is None


def test_create_legend_image_handles_numpy_palette():
    # Test that numpy arrays also work
    palette = np.array([
        [255, 255, 0],
        [0, 255, 255]
    ], dtype=np.uint8)

    img = legend.create_legend_image(palette, font_size=10, swatch_size=15, padding=2)
    assert isinstance(img, Image.Image)
    assert img.size[1] == 15 + (2 * 2)
