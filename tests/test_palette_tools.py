from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pbn import palette_tools, file_utils


def create_dummy_image():
    img = Image.new("RGB", (256, 256), color=(150, 120, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(50, 50), (150, 150)], fill=(200, 50, 50))
    draw.ellipse([(100, 100), (200, 200)], fill=(50, 200, 50))
    return img


def test_extract_palette_and_save_pbn_png(tmp_path):
    # Create dummy image
    img = create_dummy_image()

    # Save the dummy image temporarily (use BytesIO or tmp_path)
    tmp_input_path = tmp_path / "dummy_input.png"
    img.save(tmp_input_path)

    # Extract palette
    max_colors = 3
    palette = palette_tools.extract_palette_from_image(str(tmp_input_path), max_colors=max_colors)

    assert palette.shape == (max_colors, 3)
    assert palette.dtype == np.uint8

    # Validate approximate expected colors
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

    # Use file_utils to save the image with metadata (actual file output)
    output_path = tmp_path / "test_output.png"
    file_utils.save_pbn_png(
        img,
        output_path,
        command_line_invocation="pytest test",
        additional_metadata={"Test": "Palette extraction"}
    )

    # Check that the saved file exists
    assert output_path.exists()

    # Check PNG metadata
    with Image.open(output_path) as saved_img:
        info = saved_img.info
        assert "pbngen:Test" in info
        assert info["pbngen:Test"] == "Palette extraction"

