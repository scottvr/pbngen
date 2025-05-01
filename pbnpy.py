import typer
from pbn import quantize, segment, legend, stylize, palette_tools, vector_output
import os
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Optional

app = typer.Typer(help="Paint-by-number generator: quantize, segment, label, and output annotated image + palette legend.")
import rich.traceback
rich.traceback.install(show_locals=False, suppress=[__name__])

@app.command()
def generate(
    input_path: str = typer.Argument(..., help="Input image file."),
    outdir: Path = typer.Option(..., "--output_dir", "--outdir", help="Directory for output files."),
    complexity: str = typer.Option(None, help="Preset complexity level: beginner, intermediate, master."),
    style: str = typer.Option(None, help="Optional style to apply before quantization: blur, mosaic, pixelate."),
    num_colors: int = typer.Option(None, help="Number of colors in quantized output."),
    palette_from: str = typer.Option(None, help="Path to image to extract fixed palette from."),
    font_path: str = typer.Option(None, help="Path to a .ttf font file."),
    font_size: int = typer.Option(None, help="Font size for overlay labels."),
    label_mode: str = typer.Option("diagonal", help="Labeling strategy: diagonal, centroid, stable."),
    tile_spacing: int = typer.Option(None, help="Distance between repeated numbers in a region."),
    swatch_size: int = typer.Option(40, help="Width/height of each color swatch in the legend."),
    legend_height: int = typer.Option(80, help="Height in pixels of the palette legend image."),
    skip_legend: bool = typer.Option(False, help="If set, skips generating the palette legend."),
    raster_only: bool = typer.Option(False, help="Only generate raster output; skip vectorization"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Allow overwriting existing files."),
    blobbify: bool = typer.Option(False, help="Enable painterly splitting of regions into blobs."),
    blob_min: int = typer.Option(3, help="Minimum blob size in mm^2 (converted to pixels)."),
    blob_max: int = typer.Option(30, help="Maximum blob size in mm^2 (converted to pixels)."),
    min_label_font: int = typer.Option(8, help="Minimum font size allowed for blob labeling."),
    interpolate_contours: bool = typer.Option(True, help="Interpolate contour lines (useful for raster, but can be skipped in vector)."),
       dpi: int = typer.Option(None, help="DPI to use for mm² → px² conversion when blobbifying. Overrides embedded DPI if provided.")
):
    output_dir = outdir
    os.makedirs(output_dir, exist_ok=True)
    output_paths = validate_output_dir(output_dir, overwrite=yes,  
                    expect=["quantized", "raster", "legend"] if raster_only else None)

    quantized_path = output_paths["quantized"]
    vector_path = output_paths["vector"]
    legend_path = output_paths["legend"]
    raster_path = output_paths["raster"]

    presets = {
        "beginner": dict(num_colors=6, tile_spacing=40, font_size=14),
        "intermediate": dict(num_colors=12, tile_spacing=30, font_size=12),
        "master": dict(num_colors=24, tile_spacing=20, font_size=10),
    }
    if complexity in presets:
        preset = presets[complexity]
        num_colors = num_colors or preset["num_colors"]
        tile_spacing = tile_spacing or preset["tile_spacing"]
        font_size = font_size or preset["font_size"]

    num_colors = num_colors or 12
    tile_spacing = tile_spacing or 30
    font_size = font_size or 12

    styled_path = str(Path(output_dir) / "styled.png")
    quantized_path = str(Path(output_dir) / "quantized.png")
    labeled_path = str(Path(output_dir) / "labeled.png")
    legend_path = str(Path(output_dir) / "legend.png")

    input_to_use = input_path
    if style:
        input_to_use = styled_path
        stylize.apply_style(input_path, styled_path, style)
        typer.echo(f"Applied style '{style}' and saved to: {styled_path}")

    fixed_palette = None
    if palette_from:
        fixed_palette = palette_tools.extract_palette_from_image(palette_from, max_colors=num_colors)
        typer.echo(f"Extracted palette from '{palette_from}' with up to {num_colors} colors.")

    palette = quantize.quantize_image(input_to_use, quantized_path, num_colors=num_colors, fixed_palette=fixed_palette)
    typer.echo(f"Saved quantized image to: {quantized_path}")

    img_data = np.array(Image.open(quantized_path).convert("RGB"))
    canvas_size = (img_data.shape[1], img_data.shape[0])
    print(f"Canvas size (for SVG): {canvas_size}")

    primitives = segment.collect_region_primitives(
        input_path=quantized_path,
        palette=palette,
        font_size=font_size,
        font_path=font_path,
        tile_spacing=tile_spacing,
        label_mode=label_mode,
        interpolate_contours=interpolate_contours
    )

    if blobbify:
        px_per_mm = 96 / 25.4
        area_min_px = int(blob_min * px_per_mm ** 2)
        area_max_px = int(blob_max * px_per_mm ** 2)
        primitives = segment.blobbify_primitives(
            primitives,
            img_shape=img_data.shape[:2],
            min_blob_area=area_min_px,
            max_blob_area=area_max_px,
            min_label_font_size=min_label_font,
            interpolate_contours=interpolate_contours
        )
        typer.echo(f"Blobbified regions using {blob_min}-{blob_max} mm² area thresholds.")

    if not raster_only:
        vector_output.write_svg(
            output_path=vector_path,
            canvas_size=canvas_size,
            primitives=primitives
        )
        typer.echo(f"Saved vector-labeled output to: {vector_path}")

    labeled_img = segment.render_raster_from_primitives(
        canvas_size=canvas_size,
        primitives=primitives,
        font_path=font_path
    )
    labeled_img.save(labeled_path)
    typer.echo(f"Saved labeled image to: {labeled_path}")

    if not skip_legend:
        legend.generate_legend(
            palette=palette,
            output_path=legend_path,
            font_path=font_path,
            font_size=font_size,
            swatch_size=swatch_size,
            padding=10
        )
        typer.echo(f"Saved palette legend to: {legend_path}")

def validate_output_dir(
    output_dir: Path,
    overwrite: bool = False,
    expect: Optional[list] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "quantized": output_dir / "quantized.png",
        "vector": output_dir / "vector.svg",
        "legend": output_dir / "legend.png",
        "raster": output_dir / "labeled.png",
    }
    if expect:
        output_files = {k: v for k, v in output_files.items() if k in expect}
    if not overwrite:
        clobbered = [str(path) for path in output_files.values() if path.exists()]
        if clobbered:
            typer.echo("Error: The following files already exist:")
            for path in clobbered:
                typer.echo(f"  {path}")
            typer.echo("Use --yes to allow overwriting.")
            raise typer.Exit(code=1)
    return output_files

if __name__ == "__main__":
    app()
