import typer
from pbn import quantize, segment, legend, stylize, palette_tools, vector_output
import os
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
from typing import Optional

app = typer.Typer(
    help="A Paint-by-Number (PBN) generator.",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False
)

import rich.traceback
rich.traceback.install(show_locals=False, suppress=[__name__])

# Helper function for Pillow quantization (remains the same)
def quantize_pil_image(image_pil: Image.Image, num_quant_colors: int, method=Image.Quantize.MEDIANCUT) -> Image.Image:
    if image_pil.mode not in ('P', 'L'):
        image_pil = image_pil.convert('RGB')
    if image_pil.mode == 'P':
        image_pil = image_pil.convert('RGB')
    return image_pil.quantize(colors=num_quant_colors, method=method)


@app.callback(invoke_without_command=True)
def pbn_cli(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        ...,
        help="Input image file (e.g., image.jpg).",
        metavar="INPUT_FILE",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory for output files (e.g., ./output_data). Will be created if it doesn't exist.",
        metavar="OUTPUT_DIRECTORY",
        file_okay=False, dir_okay=True, writable=True, resolve_path=True,
    ),
    # --- General Options ---
    complexity: Optional[str] = typer.Option(
        None, help="Preset complexity level: beginner, intermediate, master."
    ),
    num_colors: Optional[int] = typer.Option(
        None, "--num-colors", help="Final number of colors for the PBN palette. Default: 12."
    ),
    bpp: Optional[int] = typer.Option(
        None, "--bpp",
        help="Bits Per Pixel (1-8) for an initial color depth reduction. Applied before final PBN quantization."
    ),
    palette_from: Optional[Path] = typer.Option(
        None, "--palette-from", help="Path to image to extract fixed palette from.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    # --- Style Options ---
    style: Optional[str] = typer.Option(
        None, "--style", help="Optional style to apply: blur, pixelate, mosaic."
    ),
    blur_radius: Optional[int] = typer.Option(
        None, "--blur-radius", min=1,
        help="Radius for Gaussian blur if --style is 'blur'. Default: 4 (from stylize module)."
    ),
    pixelate_block_size: Optional[int] = typer.Option(
        None, "--pixelate-block-size", min=1,
        help="Block size (pixels) for 'pixelate' style. Default: dynamic (approx image_min_dim/64, min 4)."
    ),
    mosaic_block_size: Optional[int] = typer.Option(
        None, "--mosaic-block-size", min=1,
        help="Block size (pixels) for 'mosaic' style. Default: dynamic (approx image_min_dim/64, min 4)."
    ),
    # --- Font and Label Options ---
    font_path: Optional[Path] = typer.Option(
        None, "--font-path", help="Path to a .ttf font file.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    font_size: Optional[int] = typer.Option(None, "--font-size", help="Base font size. Default: 12."),
    label_mode: str = typer.Option("diagonal", "--label-mode", help="Labeling strategy. Default: diagonal."),
    tile_spacing: Optional[int] = typer.Option(None, "--tile-spacing", help="Label distance. Default: 30px."),
    # --- Legend Options ---
    swatch_size: int = typer.Option(40, "--swatch-size", help="Legend swatch size. Default: 40px."),
    skip_legend: bool = typer.Option(False, "--skip-legend", help="Skip generating palette legend."),
    # --- Output and Operational Options ---
    raster_only: bool = typer.Option(False, "--raster-only", help="Skip vector SVG output."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite existing files."),
    interpolate_contours: bool = typer.Option(True, "--interpolate-contours/--no-interpolate-contours", help="Smooth contour lines. Default: True."),
    dpi: Optional[int] = typer.Option(None, "--dpi", help="DPI for mm² to px² conversion. Default: Auto or 96."),
    # --- Blobbify Options ---
    blobbify: bool = typer.Option(False, "--blobbify", help="Split regions into smaller 'blobs'."),
    blob_min: int = typer.Option(3, "--blob-min", help="Min blob area in mm² (if blobbify). Default: 3."),
    blob_max: int = typer.Option(30, "--blob-max", help="Max blob area in mm² (if blobbify). Default: 30."),
    min_label_font: int = typer.Option(8, "--min-label-font", help="Min font size for blob labels. Default: 8."),
):
    """
    Generates a paint-by-number set from an input image.
    """
    if ctx.invoked_subcommand is not None: return

    try:
        os.makedirs(output_dir, exist_ok=True)
        typer.echo(f"Using output directory: {output_dir}")
    except Exception as e:
        typer.secho(f"Error creating output directory {output_dir}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    expected_outputs = ["quantized", "raster", "legend"]
    if not raster_only: expected_outputs.append("vector")
    output_paths = validate_output_dir(output_dir, overwrite=yes, expect=expected_outputs)

    styled_path = output_dir / "styled_input.png"
    bpp_quantized_input_path = output_dir / "bpp_quantized_input.png"
    bpp_quantized_palette_path = output_dir / "bpp_quantized_palette_source.png"
    quantized_pbn_path = output_paths["quantized"]
    labeled_path = output_paths["raster"]
    legend_path = output_paths["legend"]
    vector_path = output_paths.get("vector")

    effective_pbn_num_colors = num_colors
    effective_tile_spacing = tile_spacing
    effective_font_size = font_size

    presets = {
        "beginner": {"num_colors": 6, "tile_spacing": 40, "font_size": 14},
        "intermediate": {"num_colors": 12, "tile_spacing": 30, "font_size": 12},
        "master": {"num_colors": 24, "tile_spacing": 20, "font_size": 10},
    }
    if complexity and complexity in presets:
        typer.echo(f"Applying preset complexity: '{complexity}'")
        preset_values = presets[complexity]
        if effective_pbn_num_colors is None: effective_pbn_num_colors = preset_values["num_colors"]
        if effective_tile_spacing is None: effective_tile_spacing = preset_values["tile_spacing"]
        if effective_font_size is None: effective_font_size = preset_values["font_size"]
    
    if effective_pbn_num_colors is None: effective_pbn_num_colors = 12
    if effective_tile_spacing is None: effective_tile_spacing = 30
    if effective_font_size is None: effective_font_size = 12
    typer.echo(f"Final PBN palette will aim for {effective_pbn_num_colors} colors.")

    current_input_image_path_for_processing: Path = input_path
    if style:
        try:
            typer.echo(f"Applying style '{style}'...")
            stylize.apply_style(
                input_path, 
                styled_path, 
                style,
                blur_radius=blur_radius, # Pass the CLI option value
                pixelate_block_size=pixelate_block_size, # Pass the CLI option value
                mosaic_block_size=mosaic_block_size # Pass the CLI option value
            )
            current_input_image_path_for_processing = styled_path
            typer.echo(f"Styled image saved to: {styled_path}")
        except ValueError as e: # Catch specific ValueErrors from stylize.py
             typer.secho(f"Styling error: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error applying style: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    path_to_main_image_for_pbn_quantization: Path = current_input_image_path_for_processing
    path_to_palette_image_for_extraction: Optional[Path] = palette_from

    if bpp is not None:
        if not (1 <= bpp <= 8):
            typer.secho(f"Error: --bpp value ({bpp}) invalid. Must be 1-8.", fg=typer.colors.RED); raise typer.Exit(code=1)
        
        num_bpp_quant_colors = 2**bpp
        typer.echo(f"Applying --bpp {bpp} pre-quantization ({num_bpp_quant_colors} colors)...")

        try:
            with Image.open(current_input_image_path_for_processing) as img_pil:
                pre_quant_input_pil = quantize_pil_image(img_pil, num_bpp_quant_colors)
                pre_quant_input_pil.save(bpp_quantized_input_path)
            path_to_main_image_for_pbn_quantization = bpp_quantized_input_path
            typer.echo(f"Pre-quantized input image saved to: {bpp_quantized_input_path}")
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            typer.secho(f"Error pre-quantizing input image {current_input_image_path_for_processing}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

        if palette_from:
            try:
                with Image.open(palette_from) as palette_img_pil:
                    pre_quant_palette_pil = quantize_pil_image(palette_img_pil, num_bpp_quant_colors)
                    pre_quant_palette_pil.save(bpp_quantized_palette_path)
                path_to_palette_image_for_extraction = bpp_quantized_palette_path
                typer.echo(f"Pre-quantized palette source image saved to: {bpp_quantized_palette_path}")
            except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
                typer.secho(f"Error pre-quantizing palette source image {palette_from}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)
    
    fixed_pbn_palette_data = None
    if path_to_palette_image_for_extraction:
        try:
            fixed_pbn_palette_data = palette_tools.extract_palette_from_image(
                path_to_palette_image_for_extraction, 
                max_colors=effective_pbn_num_colors
            )
            typer.echo(f"Extracted {len(fixed_pbn_palette_data)} colors from '{path_to_palette_image_for_extraction}' for PBN palette (max_colors: {effective_pbn_num_colors}).")
        except Exception as e:
            typer.secho(f"Error extracting PBN palette from {path_to_palette_image_for_extraction}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    try:
        final_pbn_palette = quantize.quantize_image(
            path_to_main_image_for_pbn_quantization, 
            quantized_pbn_path, 
            num_colors=effective_pbn_num_colors, 
            fixed_palette=fixed_pbn_palette_data
        )
        typer.echo(f"Final PBN quantized image saved to: {quantized_pbn_path} (using {len(final_pbn_palette)} colors).")
    except Exception as e:
        typer.secho(f"Error during final PBN quantization of {path_to_main_image_for_pbn_quantization}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    try:
        img_pil_for_segmentation = Image.open(quantized_pbn_path).convert("RGB")
        img_data_for_segmentation = np.array(img_pil_for_segmentation)
        canvas_width, canvas_height = img_pil_for_segmentation.size
        canvas_size = (canvas_width, canvas_height)
        typer.echo(f"Canvas size for segmentation: {canvas_width}x{canvas_height} pixels.")
    except Exception as e:
        typer.secho(f"Error opening PBN quantized image {quantized_pbn_path}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    primitives = segment.collect_region_primitives(
        input_path=quantized_pbn_path, palette=final_pbn_palette,
        font_size=effective_font_size, font_path=font_path,
        tile_spacing=effective_tile_spacing, label_mode=label_mode,
        interpolate_contours=interpolate_contours
    )
    typer.echo(f"Collected {len(primitives)} initial regions for labeling.")

    if blobbify:
        typer.echo("Applying blobbification...")
        effective_dpi_val = dpi
        if not effective_dpi_val:
            try:
                with Image.open(input_path) as img_orig_for_dpi:
                    dpi_info = img_orig_for_dpi.info.get('dpi')
                    if dpi_info and isinstance(dpi_info, (tuple, list)) and len(dpi_info) > 0:
                        effective_dpi_val = dpi_info[0]
                        typer.echo(f"Using DPI from input image metadata: {effective_dpi_val}")
                    else: effective_dpi_val = 96; typer.echo(f"DPI not found, defaulting to {effective_dpi_val} DPI.")
            except: effective_dpi_val = 96; typer.echo(f"Could not read DPI, defaulting to {effective_dpi_val} DPI.")
        else: typer.echo(f"Using user-provided DPI: {effective_dpi_val}")

        px_per_mm = effective_dpi_val / 25.4
        area_min_px = int(blob_min * (px_per_mm ** 2))
        area_max_px = int(blob_max * (px_per_mm ** 2))
        typer.echo(f"Blobbify settings: min={area_min_px}px² ({blob_min}mm²), max={area_max_px}px² ({blob_max}mm²) at {effective_dpi_val} DPI.")

        primitives = segment.blobbify_primitives(
            primitives, img_shape=img_data_for_segmentation.shape[:2],
            min_blob_area=area_min_px, max_blob_area=area_max_px,
            min_label_font_size=min_label_font, interpolate_contours=interpolate_contours
        )
        typer.echo(f"Blobbification resulted in {len(primitives)} regions.")

    if not raster_only and vector_path:
        try:
            vector_output.write_svg(vector_path, canvas_size, primitives)
            typer.echo(f"Saved vector (SVG) output to: {vector_path}")
        except Exception as e: typer.secho(f"Error writing SVG output: {e}", fg=typer.colors.RED)

    try:
        labeled_img = segment.render_raster_from_primitives(canvas_size, primitives, font_path)
        labeled_img.save(labeled_path)
        typer.echo(f"Saved labeled raster (PNG) image to: {labeled_path}")
    except Exception as e: typer.secho(f"Error rendering/saving raster image: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    if not skip_legend:
        try:
            legend.generate_legend(
                final_pbn_palette, legend_path, font_path,
                effective_font_size, swatch_size, padding=10
            )
            typer.echo(f"Saved palette legend to: {legend_path}")
        except Exception as e: typer.secho(f"Error generating legend: {e}", fg=typer.colors.RED)

    typer.secho("\nProcessing complete!", fg=typer.colors.GREEN)
    if output_dir:
        typer.echo(f"All outputs are in directory: {output_dir.resolve()}")


def validate_output_dir(
    output_dir: Path, overwrite: bool = False, expect: Optional[list[str]] = None,
) -> dict[str, Path]:
    default_file_names = {
        "quantized": "quantized_pbn.png",
        "vector": "vector.svg",
        "legend": "legend.png",
        "raster": "labeled.png",
        "styled_input": "styled_input.png",
        "bpp_quantized_input": "bpp_quantized_input.png",
        "bpp_quantized_palette_source": "bpp_quantized_palette_source.png"
    }
    final_output_keys = ["quantized", "vector", "legend", "raster"]
    files_to_check_for_clobber: list[Path] = []
    if expect:
        for key in expect:
            if key in default_file_names:
                 files_to_check_for_clobber.append(output_dir / default_file_names[key])
    
    if not overwrite and files_to_check_for_clobber:
        clobbered_files_found = [str(p) for p in files_to_check_for_clobber if p.exists()]
        if clobbered_files_found:
            typer.secho("Error: The following final output files already exist:", fg=typer.colors.RED)
            for path_str in clobbered_files_found: typer.secho(f"  {path_str}", fg=typer.colors.RED)
            typer.secho("Use --yes (-y) to allow overwriting.", fg=typer.colors.YELLOW); raise typer.Exit(code=1)

    all_defined_output_paths = {key: output_dir / name for key, name in default_file_names.items()}
    return all_defined_output_paths

if __name__ == "__main__":
    app()
