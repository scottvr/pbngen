# PbnPy: Python Paint-By-Number Generator

**PbnPy** is a command-line tool that converts any image into a printable, paint-by-number guide. It reduces images to a fixed color palette, segments the result into paintable regions, overlays numeric labels, and generates a color swatch legend for reference.

## Features
- Supports style preprocessing (e.g. blur, pixelate, mosaic)
- Complexity presets for beginner to master-level outputs
- Tiled labeling for paintability (not just centroids)
- Custom font support for overlays and legend
- Outputs:
  - Quantized image
  - Labeled image
  - Palette legend

## Installation
Requires Python 3.7+
```bash
pip install -r requirements.txt
```

Dependencies:
- `Pillow`
- `typer`
- `numpy`
- `scikit-learn`
- `scipy`
- `scikit-image`

## Usage
```bash
python -m pbn generate input.jpg --output-dir ./out \
  --complexity intermediate \
  --style pixelate \
  --font ./fonts/DejaVuSansMono.ttf
```

### Example Output:
- `out/quantized.png` — color-reduced version of the input
- `out/labeled.png` — indexed paint-by-number with repeated digits
- `out/legend.png` — legend bar with palette swatches and numbers

## Complexity Presets
| Name        | Colors | Tile Spacing | Font Size |
|-------------|--------|---------------|------------|
| `beginner`  | 6      | 40px          | 14         |
| `intermediate` | 12   | 30px          | 12         |
| `master`    | 24     | 20px          | 10         |

## Style Options
| Name       | Description                                 |
|------------|---------------------------------------------|
| `blur`     | Gaussian blur before quantizing             |
| `pixelate` | Chunky low-res pixelation                   |
| `mosaic`   | Mosaic effect with smoother blending        |

## Roadmap
- Vector/SVG export
- Paint-tray palette matching (`--palette-from`)
- Printable layout generator
- Artistic modes (`--style impressionist`, etc.)

## License
MIT
