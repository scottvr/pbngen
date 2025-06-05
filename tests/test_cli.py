# tests/test_cli.py
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw


def create_dummy_image(path: Path):
    img = Image.new("RGB", (256, 256), color=(150, 120, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(50, 50), (150, 150)], fill=(200, 50, 50))
    draw.ellipse([(100, 100), (200, 200)], fill=(50, 200, 50))
    img.save(path)


def test_pbngen_cli_with_all_outputs(tmp_path):
    # Create dummy input image
    input_image = tmp_path / "dummy_input.png"
    create_dummy_image(input_image)

    # Output directory (pbngen uses positional args: inputfile, outputdir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run the CLI
    result = subprocess.run(
        [
            "python", "pbngen.py",
            str(input_image),
            str(output_dir),
            "--num-colors", "3",
        ],
        capture_output=True,
        text=True
    )

    # Check that CLI ran successfully
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Expected output file basenames
    expected_files = [
        "vector-pbn_canvas.svg",
        "raster-pbn_canvas.png",
        "pbn_guide-ncolor_quantized.png",
        "palette-pbn_legend.png",
    ]

    for filename in expected_files:
        file_path = output_dir / filename
        assert file_path.exists(), f"Expected output file not found: {file_path}"

    # Optional: check stdout for success message
    assert "CuPy" in result.stdout or "Completed" in result.stdout


def test_pbngen_cli_help_output():
    # Run pbngen.py with --help to verify no crash
    result = subprocess.run(
        ["python", "pbngen.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
