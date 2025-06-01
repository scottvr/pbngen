import typer
from pbn import quantize, segment, legend, stylize, palette_tools, vector_output, file_utils # Assuming pbn.segment exists
import os
from pathlib import Path
import numpy as np # Should be fine, as xp is aliased later
from PIL import Image, UnidentifiedImageError, ImageOps
import re
from typing import Optional, List, Tuple # Added List, Tuple
import traceback

import rich.traceback
try:
    import cupy as xp
    if xp.cuda.is_available():
        print("CuPy found, using GPU in pbngen.py.")
        GPU_ENABLED = True # Ensure this is defined for pbngen.py context if needed elsewhere
    else:
        raise ImportError("CuPy found but CUDA not available")
except ImportError:
    print("CuPy not found or not usable, falling back to NumPy/SciPy for CPU in pbngen.py.")
    import numpy as xp
    GPU_ENABLED = False

from enum import Enum

class PBNFile(Enum):
    VECTOR_OUTPUT = "vector_output"
    RASTER_OUTPUT = "raster_output"
    QUANTIZED_GUIDE = "quantized_guide"
    PALETTE_LEGEND = "palette_legend"
    FILTERED_INPUT = "filtered_input"
    BPP_QUANTIZED_INPUT = "bpp_quantized_input"
    BPP_QUANTIZED_PALETTE_INPUT = "bpp_quantized_palette_input"
    CANVAS_SCALED_INPUT = "canvas_scaled_input"

def quantize_pil_image(image_pil: Image.Image, num_quant_colors: int, method=Image.Quantize.MEDIANCUT) -> Image.Image:
    if image_pil.mode not in ('P', 'L'):
        image_pil = image_pil.convert('RGB')
    if image_pil.mode == 'P': # If it was already palettized but not RGB
        image_pil = image_pil.convert('RGB') # Convert to RGB before quantizing to ensure consistency
    return image_pil.quantize(colors=num_quant_colors, method=method)


def parse_canvas_size_to_pixels(size_str: str, dpi: float) -> Optional[Tuple[int, int]]:
    match = re.fullmatch(r"([\d.]+)\s*x\s*([\d.]+)\s*(in|cm|mm)", size_str.lower())
    if not match:
        typer.secho(f"Error: Invalid --canvas-size format: '{size_str}'. "
                    "Expected format like '10x8in', '29.7x21cm', or '200x300mm'.",
                    fg=typer.colors.RED)
        return None
    try:
        width, height, unit = float(match.group(1)), float(match.group(2)), match.group(3)
    except ValueError:
        typer.secho(f"Error: Invalid numeric values in --canvas-size: '{size_str}'.", fg=typer.colors.RED)
        return None
    if unit == "cm": width_in, height_in = width / 2.54, height / 2.54
    elif unit == "mm": width_in, height_in = width / 25.4, height / 25.4
    elif unit == "in": width_in, height_in = width, height
    else: typer.secho(f"Error: Unknown unit '{unit}'.", fg=typer.colors.RED); return None
    if width_in <= 0 or height_in <= 0:
        typer.secho(f"Error: Dimensions must be positive.", fg=typer.colors.RED); return None
    target_width_px, target_height_px = round(width_in * dpi), round(height_in * dpi)
    if target_width_px < 1 or target_height_px < 1:
        typer.secho(f"Error: Calculated pixels ({target_width_px}x{target_height_px}) too small.", fg=typer.colors.RED); return None
    return target_width_px, target_height_px

def validate_output_dir(
    output_dir: Path, overwrite: bool = False, expect: Optional[List[str]] = None,
) -> dict[str, Path]:
    final_output_keys_to_check = ["quantized_guide", "vector_output", "palette_legend", "raster_output"]
    files_to_check_for_clobber: List[Path] = []
    if expect:
        for key in expect:
            if key in file_utils.default_file_names and key in final_output_keys_to_check:
                 files_to_check_for_clobber.append(output_dir / file_utils.default_file_names[key])
    if not overwrite and files_to_check_for_clobber:
        clobbered_files_found = [str(p) for p in files_to_check_for_clobber if p.exists()]
        if clobbered_files_found:
            typer.secho("Error: Files already exist:", fg=typer.colors.RED)
            for path_str in clobbered_files_found: typer.secho(f"  {path_str}", fg=typer.colors.RED)
            typer.secho("Use --yes (-y) to overwrite.", fg=typer.colors.YELLOW); raise typer.Exit(code=1)
    return {key: output_dir / name for key, name in file_utils.default_file_names.items()}


def pbn_cli(
    input_path: Path = typer.Argument(
        ...,
        help="Input image file (e.g., image.jpg).",
        metavar="INPUT_FILE",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory for output files. Will be created if it doesn't exist.",
        metavar="OUTPUT_DIRECTORY",
        file_okay=False, dir_okay=True, writable=True, resolve_path=True,
    ),
    # --- General Options ---
    preset: Optional[str] = typer.Option(
        None, help="Preset complexity level: beginner, intermediate, master."
    ),
    num_colors: Optional[int] = typer.Option(
        None, "--num-colors", help="Final number of colors for the PBN palette. Default: 12."
    ),
    dither: bool = typer.Option(
        False, "--dither", help="Enable dithering in final PBN quantization. Default: False."
    ),
    bpp: Optional[int] = typer.Option(
        None, "--bpp",
        help="Bits Per Pixel (1-8) for an initial color depth reduction. Applied before final PBN quantization."
    ),
    palette_from: Optional[Path] = typer.Option(
        None, "--palette-from", help="Path to image to extract fixed palette from.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    frequency_sort_palette: bool = typer.Option(
        True,
        "--frequency-sort-palette/--no-frequency-sort-palette",
        help="Sort the PBN palette by color frequency (most used first). Default: True."
    ),
    canvas_size_str: Optional[str] = typer.Option(
        None, "--canvas-size",
        help="Desired physical canvas size (e.g., '10x8in', '29.7x21cm'). Uses DPI from --dpi."
    ),
    dpi: Optional[int] = typer.Option(
        None, "--dpi",
        help="Target Dots Per Inch for --canvas-size and --blobbify. Default: Image metadata or 96."
    ),
    min_region_area_cli: Optional[int] = typer.Option(
        None, "--min-region-area", min=1,
        help="Minimum pixel area for a color region to be processed and labeled. Default: 50 (from segment module)."
    ),
    # --- Style Options ---
    # MODIFIED: filter type is now List[str] to accept multiple --filter options
    filter: Optional[List[str]] = typer.Option( # <--- MODIFIED HERE
        None, "--filter", help="Optional filter(s) to apply sequentially: blur, pixelate, mosaic, painterly-[lo,med,hi], smooth, smooth_more. Can be specified multiple times."
    ),
    blur_radius: Optional[int] = typer.Option(
        None, "--blur-radius", min=1, help="Radius for 'blur' filter. Default: 4"
    ),
    edge_strength: Optional[float] = typer.Option(
        None, "--edge-strength", min=0.0, max=1.0, help="Strength of Edge Enhancement (0.0-1.0). Default: 0.5"
    ),
    focus: Optional[int] = typer.Option(
        None, "--focus", min=1, help="How \"tidy\" brush strokes are for some filters. (default: 1)"
    ),
    fervor: Optional[int] = typer.Option(
        None, "--fervor", min=1, help="How \"manic\" brush strokes are for some filters. (default: 1)"
    ),
    brush_size: Optional[int] = typer.Option(
        None, "--brush-size", min=1, help="Base brush diameter for some filters."
    ),
    brush_step: Optional[int] = typer.Option(
        None, "--brush-step", min=1, help="Brush size increment for iterative filters."
    ),
    num_brushes: Optional[int] = typer.Option(
        None, "--num-brushes", min=1, help="Number of passes for iterative brush filters."
    ),
    pixelate_block_size: Optional[int] = typer.Option(
        None, "--pixelate-block-size", min=1, help="Block size for 'pixelate' filter. Default: dynamic."
    ),
    mosaic_block_size: Optional[int] = typer.Option(
        None, "--mosaic-block-size", min=1, help="Block size for 'mosaic' filter. Default: dynamic."
    ),
    # --- Font and Label Options ---
    font_path: Optional[Path] = typer.Option(
        None, "--font-path", help="Path to a .ttf font file.",
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
    ),
    font_size: Optional[int] = typer.Option(None, "--font-size", min=1, help="Base font size. Default: 10."),
    label_strategy: Optional[str] = typer.Option(None, "--label-strategy", help="Labeling strategy. (diagonal (\"quincunx\"), centroid, stable). Default: diagonal."),
    tile_spacing: Optional[int] = typer.Option(None, "--tile-spacing", min=1, help="Label distance for diagonal strategy. Default: 20px."),
    small_region_label_strategy: Optional[str] = typer.Option(
    "stable", "--small-region-strategy",
    help="Labeling strategy for small regions: stable, centroid, none. Default: stable."
    ),
    min_scaling_font_size_cli: Optional[int] = typer.Option(
        None, "--min-scaling-font-size", min=4,
        help="Minimum font size for labels when iterative scaling is applied (stable strategy). Default: 6."
    ),
    enable_stable_font_scaling_cli: bool = typer.Option(
        False,
        "--enable-stable-font-scaling",
        help="Enable iterative font scaling for 'stable' label placement strategy."
    ),
    label_color: Optional[str] = typer.Option(
        "#000000", "--label-color", help="Color for text labels in SVG and raster (e.g., '#FF0000' or 'red'). Default: '#000000' (black)."
    ),
    outline_color_cli: str = typer.Option(
        "#88ddff",
        "--outline-color",
        help="Color for PBN outlines in SVG and raster (e.g., '#88ddff' or 'lightblue'). Default: '#88ddff'."
    ),
    # --- Legend Options ---
    swatch_size: int = typer.Option(40, "--swatch-size", min=10, help="Legend swatch size. Default: 40px."),
    skip_legend: bool = typer.Option(False, "--skip-legend", help="Skip generating palette legend."),
    # --- Output and Operational Options ---
    raster_only: bool = typer.Option(False, "--raster-only", help="Skip vector SVG output."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite existing files."),
    interpolate_contours: bool = typer.Option(True, "--interpolate-contours/--no-interpolate-contours", help="Smooth contour lines. Default: True."),
    # --- Blobbify Options ---
    blobbify: bool = typer.Option(False, "--blobbify", "--blobify", help="Split regions into 'blobs'."),
    blob_min: int = typer.Option(3, "--blob-min", min=1, help="Min blob area in mm². Default: 3."),
    blob_max: int = typer.Option(30, "--blob-max", min=1, help="Max blob area in mm². Default: 30."),
    min_label_font: int = typer.Option(8, "--min-label-font", min=1, help="Font size for blob labels (fixed, not scaled). Default: 8."),
    # --- Extra/Output Options ---
    no_cruft: bool = typer.Option(
        False, "--no-cruft",
        help="Delete intermediate byproduct files upon completion."
    ),
):
    """
    Generates a paint-by-number set from an input image.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        typer.echo(f"Using output directory: {output_dir}")
    except Exception as e:
        typer.secho(f"Error creating output directory {output_dir}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    expected_outputs = ["quantized_guide", "raster_output", "palette_legend"]
    if not raster_only: expected_outputs.append("vector_output")
    output_paths = validate_output_dir(output_dir, overwrite=yes, expect=expected_outputs)

    filtered_path = output_paths["filtered_input"] # This will store the *final* output of the filter chain
    bpp_quantized_input_path = output_paths["bpp_quantized_input"]
    bpp_quantized_palette_path = output_paths["bpp_quantized_palette_input"]
    canvas_scaled_input_path = output_paths["canvas_scaled_input"]
    quantized_pbn_path = output_paths["quantized_guide"]
    labeled_path = output_paths["raster_output"]
    legend_path = output_paths["palette_legend"]
    vector_path = output_paths.get("vector_output")

    effective_pbn_num_colors = num_colors
    effective_tile_spacing = tile_spacing
    effective_font_size = font_size
    effective_label_strategy = label_strategy

    presets = {
        "beginner": {"num_colors": 6, "tile_spacing": 30, "font_size": 10, "label_strategy": "diagonal"},
        "intermediate": {"num_colors": 12, "tile_spacing": 20, "font_size": 10, "label_strategy": "stable"},
        "master": {"num_colors": 24, "tile_spacing": 10, "font_size": 12, "label_strategy": "stable"},
    }
    if preset and preset in presets:
        typer.echo(f"Applying preset complexity: '{preset}'")
        preset_values = presets[preset]
        if effective_pbn_num_colors is None: effective_pbn_num_colors = preset_values["num_colors"]
        if effective_tile_spacing is None: effective_tile_spacing = preset_values["tile_spacing"]
        if effective_font_size is None: effective_font_size = preset_values["font_size"]
        if effective_label_strategy is None: effective_label_strategy = preset_values["label_strategy"]
    if effective_pbn_num_colors is None: effective_pbn_num_colors = 12
    if effective_tile_spacing is None: effective_tile_spacing = 20
    if effective_font_size is None: effective_font_size = 10

    typer.echo(f"Final PBN palette will aim for {effective_pbn_num_colors} colors.")
    typer.echo(f"Target font size for labels: {effective_font_size}.")
    if not raster_only:
        typer.echo(f"Target SVG label color: {label_color}")
    if enable_stable_font_scaling_cli:
        typer.echo("Iterative font scaling for 'stable' labels is ENABLED.")
    else:
        typer.echo("Iterative font scaling for 'stable' labels is DISABLED.")

    effective_dpi_val = dpi
    if not effective_dpi_val:
        try:
            with Image.open(input_path) as img_orig_for_dpi:
                dpi_info = img_orig_for_dpi.info.get('dpi')
                if dpi_info and isinstance(dpi_info, (tuple, list)) and len(dpi_info) > 0 and dpi_info[0] > 0:
                    effective_dpi_val = int(dpi_info[0]); typer.echo(f"Using DPI from input image metadata: {effective_dpi_val}")
                else: effective_dpi_val = 96; typer.echo(f"DPI not found or invalid in image, defaulting to {effective_dpi_val} DPI.")
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            effective_dpi_val = 96; typer.echo(f"Could not read DPI from image ({e}), defaulting to {effective_dpi_val} DPI.")
    else: typer.echo(f"Using user-provided DPI: {effective_dpi_val}")

    # --- MODIFIED: Filter application logic for chaining ---
    current_input_image_path_for_processing: Path = input_path
    processed_through_filter_chain = False
    intermediate_filter_files_to_clean: List[Path] = []

    if filter and len(filter) > 0 : # filter is now List[str]
        typer.echo(f"Starting filter chain with {len(filter)} filter(s)...")
        temp_input_for_current_filter_step = current_input_image_path_for_processing
        
        for i, filter_name in enumerate(filter):
            is_last_filter = (i == len(filter) - 1)
            
            if is_last_filter:
                # The final output of the chain goes to the standard 'filtered_path'
                output_for_this_filter_step = filtered_path 
            else:
                # Intermediate outputs get unique names and will be PNGs
                output_for_this_filter_step = (output_dir / f"_intermediate_filter_step_{i}").with_suffix(".png")
                intermediate_filter_files_to_clean.append(output_for_this_filter_step)

            typer.echo(f"  Applying filter {i+1}/{len(filter)}: '{filter_name}'...")
            typer.echo(f"    Input: {temp_input_for_current_filter_step.name}")
            typer.echo(f"    Output: {output_for_this_filter_step.name}")
            try:
                stylize.apply_filter(
                    str(temp_input_for_current_filter_step), # stylize.py expects string paths
                    str(output_for_this_filter_step), 
                    filter_name, 
                    blur_radius=blur_radius, 
                    edge_strength=edge_strength,
                    pixelate_block_size=pixelate_block_size, 
                    mosaic_block_size=mosaic_block_size, 
                    num_brushes=num_brushes, 
                    brush_size=brush_size, 
                    brush_step=brush_step, 
                    focus=focus, 
                    fervor=fervor
                )
                temp_input_for_current_filter_step = output_for_this_filter_step
                processed_through_filter_chain = True 
                # typer.echo(f"  Intermediate filtered image saved to: {output_for_this_filter_step}") # Path is already printed
            except ValueError as e: 
                typer.secho(f"Styling error with filter '{filter_name}': {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)
            except Exception as e: 
                typer.secho(f"Unexpected error applying filter '{filter_name}': {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)
        
        if processed_through_filter_chain:
            current_input_image_path_for_processing = temp_input_for_current_filter_step # This will be the final filtered output path
            typer.echo(f"Filter chain complete. Final styled image for next steps: {current_input_image_path_for_processing.name}")
    # --- End of MODIFIED filter application logic ---

    path_to_main_image_for_canvas_scaling: Path = current_input_image_path_for_processing
    path_to_palette_image_for_extraction: Optional[Path] = palette_from
    if bpp is not None:
        if not (1 <= bpp <= 8): typer.secho(f"Error: --bpp ({bpp}) invalid. Must be 1-8.", fg=typer.colors.RED); raise typer.Exit(code=1)
        num_bpp_quant_colors = 2**bpp
        typer.echo(f"Applying --bpp {bpp} pre-quantization ({num_bpp_quant_colors} colors)...")
        try:
            with Image.open(current_input_image_path_for_processing) as img_pil: # This is after filtering if any
                pre_quant_input_pil = quantize_pil_image(img_pil, num_bpp_quant_colors)
                pre_quant_input_pil.save(bpp_quantized_input_path)
            path_to_main_image_for_canvas_scaling = bpp_quantized_input_path
            typer.echo(f"Pre-quantized input image saved to: {bpp_quantized_input_path}")
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            typer.secho(f"Error pre-quantizing input {current_input_image_path_for_processing}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)
        
        if palette_from:
            try:
                with Image.open(palette_from) as palette_img_pil:
                    pre_quant_palette_pil = quantize_pil_image(palette_img_pil, num_bpp_quant_colors)
                    pre_quant_palette_pil.save(bpp_quantized_palette_path)
                path_to_palette_image_for_extraction = bpp_quantized_palette_path
                typer.echo(f"Pre-quantized palette source saved to: {bpp_quantized_palette_path}")
            except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
                typer.secho(f"Error pre-quantizing palette source {palette_from}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    path_to_main_image_for_pbn_quantization: Path = path_to_main_image_for_canvas_scaling
    output_canvas_dimensions_px: Optional[Tuple[int, int]] = None
    if canvas_size_str:
        dpi_for_canvas = effective_dpi_val
        typer.echo(f"Targeting canvas size '{canvas_size_str}' using DPI {dpi_for_canvas}.")
        if dpi_for_canvas < 150 and not dpi:
             typer.secho(f"Note: DPI for canvas scaling is {dpi_for_canvas}. For print, consider a higher --dpi (e.g., 300).", fg=typer.colors.BLUE)
        parsed_target_dims_px = parse_canvas_size_to_pixels(canvas_size_str, dpi_for_canvas)
        if parsed_target_dims_px is None: raise typer.Exit(code=1)
        target_canvas_width_px, target_canvas_height_px = parsed_target_dims_px
        output_canvas_dimensions_px = (target_canvas_width_px, target_canvas_height_px)
        try:
            with Image.open(path_to_main_image_for_canvas_scaling) as img_to_scale_on_canvas: # This is after filtering and/or bpp
                img_to_scale_on_canvas_rgba = img_to_scale_on_canvas.convert("RGBA")
                source_w, source_h = img_to_scale_on_canvas_rgba.size
                if source_w == 0 or source_h == 0: typer.secho(f"Error: Source image {path_to_main_image_for_canvas_scaling} has zero dimension.", fg=typer.colors.RED); raise typer.Exit(code=1)
                
                img_aspect_ratio = source_w / source_h
                canvas_aspect_ratio = target_canvas_width_px / target_canvas_height_px
                
                if img_aspect_ratio > canvas_aspect_ratio:
                    scaled_w = target_canvas_width_px
                    scaled_h = round(scaled_w / img_aspect_ratio)
                else:
                    scaled_h = target_canvas_height_px
                    scaled_w = round(scaled_h * img_aspect_ratio)
                
                scaled_w, scaled_h = max(1, scaled_w), max(1, scaled_h)
                resized_content_img = img_to_scale_on_canvas_rgba.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                final_output_pil_canvas = Image.new("RGB", (target_canvas_width_px, target_canvas_height_px), (255, 255, 255))
                paste_x = (target_canvas_width_px - scaled_w) // 2
                paste_y = (target_canvas_height_px - scaled_h) // 2
                final_output_pil_canvas.paste(resized_content_img, (paste_x, paste_y), mask=resized_content_img if resized_content_img.mode == 'RGBA' else None)
                final_output_pil_canvas.save(canvas_scaled_input_path)
            path_to_main_image_for_pbn_quantization = canvas_scaled_input_path
            typer.echo(f"Image placed on {target_canvas_width_px}x{target_canvas_height_px}px canvas, saved to: {canvas_scaled_input_path}")
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            typer.secho(f"Error scaling image to canvas for '{path_to_main_image_for_canvas_scaling}': {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)

    fixed_pbn_palette_data = None
    if path_to_palette_image_for_extraction:
        try:
            fixed_pbn_palette_data = palette_tools.extract_palette_from_image(str(path_to_palette_image_for_extraction), max_colors=effective_pbn_num_colors)
            typer.echo(f"Extracted {len(fixed_pbn_palette_data)} colors from '{path_to_palette_image_for_extraction}' for PBN palette.")
        except Exception as e: typer.secho(f"Error extracting PBN palette from {path_to_palette_image_for_extraction}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)
    
    try:
        final_pbn_palette = quantize.quantize_image(str(path_to_main_image_for_pbn_quantization), str(quantized_pbn_path), 
                                                    num_colors=effective_pbn_num_colors, 
                                                    fixed_palette=fixed_pbn_palette_data, 
                                                    dither=dither,
                                                    sort_by_frequency=frequency_sort_palette)
        typer.echo(f"Final PBN quantized image saved to: {quantized_pbn_path} (using {len(final_pbn_palette)} colors).")
    except Exception as e: typer.secho(f"Error during PBN quantization of {path_to_main_image_for_pbn_quantization}: {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)

    canvas_size_for_final_output: Tuple[int, int]
    img_data_for_segmentation_shape: Tuple[int, int]
    try:
        with Image.open(quantized_pbn_path) as img_pil_for_segmentation:
            img_data_for_segmentation_shape = img_pil_for_segmentation.size[::-1]
            canvas_size_for_final_output = img_pil_for_segmentation.size
        
        if output_canvas_dimensions_px and output_canvas_dimensions_px != canvas_size_for_final_output:
             typer.secho(f"Internal Warning: Expected canvas dimensions {output_canvas_dimensions_px} but "
                        f"PBN quantized image is {canvas_size_for_final_output}. Using actual PBN image size for output.", fg=typer.colors.YELLOW)
        typer.echo(f"Using PBN image size for segmentation/output: {canvas_size_for_final_output[0]}x{canvas_size_for_final_output[1]} pixels (from '{quantized_pbn_path}').")
    except Exception as e: typer.secho(f"Error opening PBN quantized image {quantized_pbn_path} for size check: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    effective_min_scaling_font_size: int
    if min_scaling_font_size_cli is not None:
        effective_min_scaling_font_size = min_scaling_font_size_cli
        if enable_stable_font_scaling_cli:
            typer.echo(f"Using user-defined minimum font size for iterative scaling: {effective_min_scaling_font_size}")
    else:
        effective_min_scaling_font_size = 6 
        if enable_stable_font_scaling_cli:
            typer.echo(f"Using default minimum font size for iterative scaling: {effective_min_scaling_font_size}")
    if effective_min_scaling_font_size > effective_font_size:
        if enable_stable_font_scaling_cli:
            typer.secho(f"Warning: --min-scaling-font-size ({effective_min_scaling_font_size}) is greater than target font-size ({effective_font_size}). "
                        f"Adjusting minimum scaling font size to {effective_font_size}.", fg=typer.colors.YELLOW)
        effective_min_scaling_font_size = effective_font_size
    effective_min_scaling_font_size = max(1, effective_min_scaling_font_size)

    segment_call_kwargs = {
        "input_path": str(quantized_pbn_path), # segment.py likely expects str
        "palette": final_pbn_palette,
        "font_size": effective_font_size,
        "font_path": str(font_path) if font_path else None, # Pass str Path object
        "tile_spacing": effective_tile_spacing,
        "label_strategy": effective_label_strategy,
        "small_region_label_strategy": small_region_label_strategy,
        "interpolate_contours": interpolate_contours,
        "min_font_size_for_scaling": effective_min_scaling_font_size,
        "enable_font_scaling": enable_stable_font_scaling_cli,
    }
    if min_region_area_cli is not None:
        segment_call_kwargs["min_region_area"] = min_region_area_cli
        typer.echo(f"Using user-defined min_region_area: {min_region_area_cli} pixels.")
    
    primitives = segment.collect_region_primitives(**segment_call_kwargs) # type: ignore
    typer.echo(f"Collected {len(primitives)} regions initially.")
    typer.echo(f"Collected {sum(len(p['labels']) for p in primitives)} potential labels before collision resolution.")

    render_nudge_pixels_up = 0 
    font_path_for_collision_str = str(font_path) if font_path else None
    primitives = segment.resolve_label_collisions( # type: ignore
        primitives,
        font_path_str=font_path_for_collision_str,
        default_font_size_for_fallback=effective_font_size,
        additional_nudge_pixels_up=render_nudge_pixels_up,
    )
    typer.echo(f"Retained {sum(len(p['labels']) for p in primitives)} labels after collision resolution.")

    if blobbify:
        typer.echo("Applying blobbification...")
        px_per_mm = effective_dpi_val / 25.4
        area_min_px = int(blob_min * (px_per_mm**2))
        area_max_px = int(blob_max * (px_per_mm**2))
        typer.echo(f"Blobbify: DPI={effective_dpi_val}, min_area={area_min_px}px² ({blob_min}mm²), max_area={area_max_px}px² ({blob_max}mm²).")
        primitives = segment.blobbify_primitives(primitives, img_data_for_segmentation_shape, area_min_px, area_max_px, min_label_font, interpolate_contours) # type: ignore
        typer.echo(f"After blobbification: {len(primitives)} regions processed.")

    typer.echo(f"Target SVG label color: {label_color}")
    typer.echo(f"Target outline color: {outline_color_cli}") 

    font_path_for_vector_str = str(font_path) if font_path else None
    if not raster_only and vector_path:
        try:
            vector_output.write_svg(
                str(vector_path),
                canvas_size_for_final_output,
                primitives,
                font_path_str=font_path_for_vector_str,
                default_font_size=effective_font_size,
                label_color_str=label_color,
                outline_color_hex=outline_color_cli
            )
            typer.echo(f"SVG output saved to: {vector_path}")
        except Exception as e:
            typer.secho(f"Error writing SVG output: {e}", fg=typer.colors.RED); traceback.print_exc()    

    try:
        labeled_img = segment.render_raster_from_primitives( # type: ignore
            canvas_size_for_final_output,
            primitives,
            font_path, # render_raster_from_primitives might take Path object or str
            additional_nudge_pixels_up=render_nudge_pixels_up,
            label_text_color=label_color,
            outline_color_str_hex=outline_color_cli
        )
        labeled_img.save(labeled_path)
        typer.echo(f"Labeled raster PNG output saved to: {labeled_path}")
    except Exception as e:
        typer.secho(f"Error rendering raster output: {e}", fg=typer.colors.RED)
        traceback.print_exc()
        raise typer.Exit(code=1)

    if not skip_legend:
        try: 
            legend.generate_legend(final_pbn_palette, str(legend_path), str(font_path) if font_path else None, # type: ignore
                                   effective_font_size, swatch_size, 10)
            typer.echo(f"Palette legend saved to: {legend_path}")
        except Exception as e: typer.secho(f"Error generating palette legend: {e}", fg=typer.colors.RED); traceback.print_exc()
    
    typer.secho("\nProcessing complete!", fg=typer.colors.GREEN)

    # --- MODIFIED: No Cruft Logic ---
    if no_cruft:
        typer.echo("\n--no-cruft active: Cleaning up intermediate files...")
        
        files_to_delete_by_no_cruft: List[Path] = []
        
        # Add intermediate files from the filter chain explicitly
        files_to_delete_by_no_cruft.extend(intermediate_filter_files_to_clean)

        # Add other standard intermediate files if they were created AND are not the
        # direct input to the PBN quantization step.
        # path_to_main_image_for_pbn_quantization is the image that is actually quantized.
        
        # If filtered_path was created (i.e., filters were applied) AND
        # it's not the final image used for PBN quantization (e.g., it was further processed by BPP or canvas scaling),
        # then it's cruft.
        if processed_through_filter_chain and filtered_path.exists() and filtered_path != path_to_main_image_for_pbn_quantization:
            files_to_delete_by_no_cruft.append(filtered_path)

        # If bpp_quantized_input_path was created AND it's not the final image for PBN quantization
        if bpp is not None and bpp_quantized_input_path.exists() and bpp_quantized_input_path != path_to_main_image_for_pbn_quantization:
            files_to_delete_by_no_cruft.append(bpp_quantized_input_path)
        
        # bpp_quantized_palette_path is always cruft if it was created
        if bpp is not None and palette_from and bpp_quantized_palette_path.exists():
            files_to_delete_by_no_cruft.append(bpp_quantized_palette_path)

        # If canvas_scaled_input_path was created AND it's not the final image for PBN quantization
        # (This is rare, as canvas_scaled_input_path usually *is* path_to_main_image_for_pbn_quantization if this step ran)
        if canvas_size_str and canvas_scaled_input_path.exists() and canvas_scaled_input_path != path_to_main_image_for_pbn_quantization:
            files_to_delete_by_no_cruft.append(canvas_scaled_input_path)
        
        deleted_any_cruft = False
        unique_files_to_delete = set(files_to_delete_by_no_cruft) # Avoid trying to delete same path twice

        for file_path_to_delete in unique_files_to_delete:
            if file_path_to_delete and file_path_to_delete.exists(): # Check existence again just in case
                try:
                    file_path_to_delete.unlink()
                    typer.echo(f"  Deleted: {file_path_to_delete.name}")
                    deleted_any_cruft = True
                except OSError as e:
                    typer.secho(f"  Error deleting {file_path_to_delete.name}: {e}", fg=typer.colors.RED)
        
        if not deleted_any_cruft:
            # Check if there were any files initially identified for deletion but perhaps didn't exist
            initial_cruft_existed = any(p.exists() for p in unique_files_to_delete if p) # Ensure p is not None
            if not initial_cruft_existed and not any(intermediate_filter_files_to_clean): # if intermediate_filter_files_to_clean was empty too
                 typer.echo("  No intermediate byproduct files were found or identified for deletion.")

    if output_dir: typer.echo(f"Outputs in: {output_dir.resolve()}")

if __name__ == "__main__":
    rich.traceback.install(show_locals=False, suppress=[typer, __name__]) # type: ignore
    typer.run(pbn_cli)