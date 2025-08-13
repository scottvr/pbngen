import typer
from pbn import quantize, segment, legend, stylize, palette_tools, file_utils
import os
from pathlib import Path
import numpy as np # Should be fine, as xp is aliased later
from PIL import Image, UnidentifiedImageError, ImageOps
import re
from typing import Optional, List, Tuple, Dict # Added Dict

import traceback
import sys # Added for sys.argv

import rich.traceback
try:
    import cupy as xp
    if xp.cuda.is_available():
        print("CuPy found, using GPU in pbngen.py.")
        GPU_ENABLED = True
    else:
        raise ImportError("CuPy found but CUDA not available")
except ImportError:
    print("CuPy not found or not usable, falling back to NumPy/SciPy for CPU in pbngen.py.")
    import numpy as xp # type: ignore
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

# Map PBNFile enum members to their base filenames
PBN_FILE_BASENAMES: Dict[PBNFile, str] = {
    PBNFile.VECTOR_OUTPUT: "pbn-vector_canvas.svg",
    PBNFile.RASTER_OUTPUT: "pbn-raster_canvas.png",
    PBNFile.QUANTIZED_GUIDE: "pbn-guide_ncolor_quantized.png",
    PBNFile.PALETTE_LEGEND: "pbn-palette_legend.png",
    PBNFile.FILTERED_INPUT: "input-filtered.png",
    PBNFile.BPP_QUANTIZED_INPUT: "input-bpp_quantized.png",
    PBNFile.BPP_QUANTIZED_PALETTE_INPUT: "input-palette_bpp_quantized.png",
    PBNFile.CANVAS_SCALED_INPUT: "input-canvas_scaled.png",
}

# Enum members that represent final (non-intermediate) outputs
FINAL_OUTPUT_ENUM_KEYS: List[PBNFile] = [
    PBNFile.QUANTIZED_GUIDE,
    PBNFile.VECTOR_OUTPUT,
    PBNFile.PALETTE_LEGEND,
    PBNFile.RASTER_OUTPUT
]

def quantize_pil_image(image_pil: Image.Image, num_quant_colors: int, method=Image.Quantize.MEDIANCUT) -> Image.Image:
    if image_pil.mode not in ('P', 'L'):
        image_pil = image_pil.convert('RGB')
    if image_pil.mode == 'P':
        image_pil = image_pil.convert('RGB')
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
    else: typer.secho(f"Error: Unknown unit '{unit}'.", fg=typer.colors.RED); return None # type: ignore
    if width_in <= 0 or height_in <= 0:
        typer.secho(f"Error: Dimensions must be positive.", fg=typer.colors.RED); return None
    target_width_px, target_height_px = round(width_in * dpi), round(height_in * dpi)
    if target_width_px < 1 or target_height_px < 1:
        typer.secho(f"Error: Calculated pixels ({target_width_px}x{target_height_px}) too small.", fg=typer.colors.RED); return None
    return target_width_px, target_height_px

def validate_output_dir(
    output_dir: Path, overwrite: bool = False, expect: Optional[List[PBNFile]] = None,
) -> Dict[PBNFile, Path]:
    files_to_check_for_clobber: List[Path] = []
    if expect:
        for pbn_file_key in expect:
            if pbn_file_key in PBN_FILE_BASENAMES and pbn_file_key in FINAL_OUTPUT_ENUM_KEYS:
                 files_to_check_for_clobber.append(output_dir / PBN_FILE_BASENAMES[pbn_file_key])
    
    if not overwrite and files_to_check_for_clobber:
        clobbered_files_found = [str(p) for p in files_to_check_for_clobber if p.exists()]
        if clobbered_files_found:
            typer.secho("Error: Files already exist:", fg=typer.colors.RED)
            for path_str in clobbered_files_found: typer.secho(f"  {path_str}", fg=typer.colors.RED)
            typer.secho("Use --yes (-y) to overwrite.", fg=typer.colors.YELLOW); raise typer.Exit(code=1)
    
    return {key: output_dir / name for key, name in PBN_FILE_BASENAMES.items()}


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
    filter: Optional[List[str]] = typer.Option(  
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
    label_strategy: Optional[str] = typer.Option("diagonal", "--label-strategy", help="Labeling strategy. (diagonal (\"quincunx\"), centroid, stable). Default: diagonal."),
    render_nudge_pixels_up: Optional[int] = typer.Option(0,"--nudge-labels", help="nudge labels vertically by this many pixels offset. (negative for down, positive for up.)"),
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
        True,
        "--enable-stable-font-scaling/--no-enable-stable-font-scaling",
        help="Enable iterative font scaling for 'stable' label placement strategy."
    ),
    label_color: Optional[str] = typer.Option(
        "#88ddff", "--label-color", help="Color for text labels in SVG and raster (e.g., '#FF0000' or 'red'). Default: '#88ddff' (black)."
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
    command_line_str = " ".join(sys.argv) # Capture command line invocation

    try:
        os.makedirs(output_dir, exist_ok=True)
        typer.echo(f"Using output directory: {output_dir}")
    except Exception as e:
        typer.secho(f"Error creating output directory {output_dir}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    # UPDATED: Capture initial directory state for safe --no-cruft cleanup
    initial_files_in_dir = set(os.listdir(output_dir)) if output_dir.exists() else set()

    # Use PBNFile enum members for expected_outputs
    expected_outputs: List[PBNFile] = [
        PBNFile.QUANTIZED_GUIDE, PBNFile.RASTER_OUTPUT, PBNFile.PALETTE_LEGEND
    ]
    if not raster_only:
        expected_outputs.append(PBNFile.VECTOR_OUTPUT)
    
    output_paths: Dict[PBNFile, Path] = validate_output_dir(output_dir, overwrite=yes, expect=expected_outputs)

    # Access paths using PBNFile enum members
    filtered_path = output_paths[PBNFile.FILTERED_INPUT]
    bpp_quantized_input_path = output_paths[PBNFile.BPP_QUANTIZED_INPUT]
    bpp_quantized_palette_path = output_paths[PBNFile.BPP_QUANTIZED_PALETTE_INPUT]
    canvas_scaled_input_path = output_paths[PBNFile.CANVAS_SCALED_INPUT]
    quantized_pbn_path = output_paths[PBNFile.QUANTIZED_GUIDE]
    labeled_path = output_paths[PBNFile.RASTER_OUTPUT]
    legend_path = output_paths[PBNFile.PALETTE_LEGEND]
    vector_path = output_paths.get(PBNFile.VECTOR_OUTPUT) # .get() is fine as it might not exist if raster_only

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

    current_input_image_path_for_processing: Path = input_path
    processed_through_filter_chain = False
    last_filter_output_pil: Optional[Image.Image] = None


    if filter and len(filter) > 0 :
        typer.echo(f"Starting filter chain with {len(filter)} filter(s)...")
        temp_input_for_current_filter_step_path = current_input_image_path_for_processing # Path for the first filter
        
        for i, filter_name in enumerate(filter):
            is_last_filter = (i == len(filter) - 1)
            
            # Determine output path for this filter step
            # If it's the last filter, output to the final filtered_path.
            # Otherwise, create an intermediate file path.
            output_path_for_this_filter_step: Path
            if is_last_filter:
                output_path_for_this_filter_step = filtered_path
            else:
                # Create a unique name for intermediate filtered files
                output_path_for_this_filter_step = (output_dir / f"_intermediate_filter_step_{i}_{filter_name.replace('-', '_')}").with_suffix(".png")

            typer.echo(f"  Applying filter {i+1}/{len(filter)}: '{filter_name}'...")
            # The input to stylize.apply_filter is always a path for the first filter in a chain,
            # or the path of the previously saved intermediate filter result.
            input_to_filter_module = str(temp_input_for_current_filter_step_path)
            typer.echo(f"    Input to stylize: {input_to_filter_module}")

            try:
                # stylize.apply_filter now returns a PIL image
                filtered_pil_image = stylize.apply_filter(
                    input_to_filter_module, 
                    filter_name, # output_path is no longer passed here
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
                
                if filtered_pil_image:
                    # Save the returned PIL image using file_utils
                    file_utils.save_pbn_png(
                        filtered_pil_image,
                        output_path_for_this_filter_step,
                        command_line_invocation=command_line_str,
                        additional_metadata={
                            "PbNgen-FileType": "Intermediate Filtered Image" if not is_last_filter else "Final Filtered Input",
                            "SourceImage": str(temp_input_for_current_filter_step_path),
                            "FilterApplied": filter_name,
                            "FilterStep": f"{i+1}/{len(filter)}"
                        }
                    )
                    typer.echo(f"    Output saved: {output_path_for_this_filter_step.name}")
                    temp_input_for_current_filter_step_path = output_path_for_this_filter_step # Next filter uses this output as input
                    if is_last_filter:
                        last_filter_output_pil = filtered_pil_image # Keep PIL if it's the last one
                else:
                    # This case should ideally not happen if apply_filter raises errors properly
                    typer.secho(f"Error: Filter '{filter_name}' did not return an image.", fg=typer.colors.RED); raise typer.Exit(code=1)

                processed_through_filter_chain = True 
            except ValueError as e: 
                typer.secho(f"Styling error with filter '{filter_name}': {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)
            except Exception as e: 
                typer.secho(f"Unexpected error applying filter '{filter_name}': {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)
        
        if processed_through_filter_chain:
            current_input_image_path_for_processing = temp_input_for_current_filter_step_path # This is now a Path object
            typer.echo(f"Filter chain complete. Final styled image for next steps: {current_input_image_path_for_processing.name}")


    path_to_main_image_for_canvas_scaling: Path = current_input_image_path_for_processing
    path_to_palette_image_for_extraction: Optional[Path] = palette_from

    if bpp is not None:
        if not (1 <= bpp <= 8): typer.secho(f"Error: --bpp ({bpp}) invalid. Must be 1-8.", fg=typer.colors.RED); raise typer.Exit(code=1)
        num_bpp_quant_colors = 2**bpp
        typer.echo(f"Applying --bpp {bpp} pre-quantization ({num_bpp_quant_colors} colors)...")
        try:
            # Use current_input_image_path_for_processing which is a Path
            with Image.open(current_input_image_path_for_processing) as img_pil:
                pre_quant_input_pil = quantize_pil_image(img_pil, num_bpp_quant_colors)
                file_utils.save_pbn_png(
                    pre_quant_input_pil,
                    bpp_quantized_input_path,
                    command_line_invocation=command_line_str,
                    additional_metadata={
                        "PbNgen-FileType": "BPP Pre-quantized Input", # Using PBNFile.BPP_QUANTIZED_INPUT.value
                        "SourceImage": str(current_input_image_path_for_processing),
                        "BPP": str(bpp),
                        "QuantColors": str(num_bpp_quant_colors)
                    }
                )
            path_to_main_image_for_canvas_scaling = bpp_quantized_input_path
            typer.echo(f"Pre-quantized input image saved to: {bpp_quantized_input_path}")
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            typer.secho(f"Error pre-quantizing input {current_input_image_path_for_processing}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)
        
        if palette_from: # palette_from is already a Path or None
            try:
                with Image.open(palette_from) as palette_img_pil:
                    pre_quant_palette_pil = quantize_pil_image(palette_img_pil, num_bpp_quant_colors)
                    file_utils.save_pbn_png(
                        pre_quant_palette_pil,
                        bpp_quantized_palette_path,
                        command_line_invocation=command_line_str,
                        additional_metadata={
                            "PbNgen-FileType": "BPP Pre-quantized Palette Source", # Using PBNFile.BPP_QUANTIZED_PALETTE_INPUT.value
                            "SourcePaletteImage": str(palette_from),
                            "BPP": str(bpp),
                            "QuantColors": str(num_bpp_quant_colors)
                        }
                    )
                path_to_palette_image_for_extraction = bpp_quantized_palette_path
                typer.echo(f"Pre-quantized palette source saved to: {bpp_quantized_palette_path}")
            except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
                typer.secho(f"Error pre-quantizing palette source {palette_from}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)

    path_to_main_image_for_pbn_quantization: Path = path_to_main_image_for_canvas_scaling
    output_canvas_dimensions_px: Optional[Tuple[int, int]] = None
    if canvas_size_str:
        dpi_for_canvas = effective_dpi_val # Already an int
        typer.echo(f"Targeting canvas size '{canvas_size_str}' using DPI {dpi_for_canvas}.")
        if dpi_for_canvas < 150 and not dpi: # dpi is the user option, effective_dpi_val is the derived one
             typer.secho(f"Note: DPI for canvas scaling is {dpi_for_canvas}. For print, consider a higher --dpi (e.g., 300).", fg=typer.colors.BLUE)
        parsed_target_dims_px = parse_canvas_size_to_pixels(canvas_size_str, float(dpi_for_canvas)) # Ensure float for parse
        if parsed_target_dims_px is None: raise typer.Exit(code=1)
        target_canvas_width_px, target_canvas_height_px = parsed_target_dims_px
        output_canvas_dimensions_px = (target_canvas_width_px, target_canvas_height_px)
        try:
            with Image.open(path_to_main_image_for_canvas_scaling) as img_to_scale_on_canvas:
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
                
                file_utils.save_pbn_png(
                    final_output_pil_canvas,
                    canvas_scaled_input_path,
                    command_line_invocation=command_line_str,
                    additional_metadata={
                        "PbNgen-FileType": "Canvas Scaled Input", # Using PBNFile.CANVAS_SCALED_INPUT.value
                        "SourceImage": str(path_to_main_image_for_canvas_scaling),
                        "TargetCanvasSize": f"{target_canvas_width_px}x{target_canvas_height_px}px",
                        "EffectiveDPI": str(dpi_for_canvas)
                    }
                )
            path_to_main_image_for_pbn_quantization = canvas_scaled_input_path
            typer.echo(f"Image placed on {target_canvas_width_px}x{target_canvas_height_px}px canvas, saved to: {canvas_scaled_input_path}")
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            typer.secho(f"Error scaling image to canvas for '{path_to_main_image_for_canvas_scaling}': {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)

    fixed_pbn_palette_data = None
    if path_to_palette_image_for_extraction: # This is a Path object
        try:
            fixed_pbn_palette_data = palette_tools.extract_palette_from_image(str(path_to_palette_image_for_extraction), max_colors=effective_pbn_num_colors)
            typer.echo(f"Extracted {len(fixed_pbn_palette_data)} colors from '{path_to_palette_image_for_extraction}' for PBN palette.")
        except Exception as e: typer.secho(f"Error extracting PBN palette from {path_to_palette_image_for_extraction}: {e}", fg=typer.colors.RED); raise typer.Exit(code=1)
    
    final_pbn_palette: Optional[np.ndarray] = None # Define before try block
    try:
        # quantize.quantize_image now returns (PIL.Image, palette_array)
        quantized_pil_image, final_pbn_palette = quantize.quantize_image(
            input_path=str(path_to_main_image_for_pbn_quantization),
            # output_path is no longer an arg for quantize_image
            num_colors=effective_pbn_num_colors, 
            fixed_palette=fixed_pbn_palette_data, 
            dither=dither,
            sort_by_frequency=frequency_sort_palette
        )
        
        if quantized_pil_image and final_pbn_palette is not None:
            # Save the returned PIL image using file_utils
            file_utils.save_pbn_png(
                quantized_pil_image,
                quantized_pbn_path, # This is output_paths[PBNFile.QUANTIZED_GUIDE]
                command_line_invocation=command_line_str,
                additional_metadata={
                    "PbNgen-FileType": "PBN Quantized Guide", # Descriptive, or PBNFile.QUANTIZED_GUIDE.value
                    "SourceImage": str(path_to_main_image_for_pbn_quantization),
                    "NumColorsTarget": str(effective_pbn_num_colors),
                    "NumColorsActual": str(len(final_pbn_palette)),
                    "Dithering": str(dither),
                    "PaletteSortedByFrequency": str(frequency_sort_palette),
                    "FixedPaletteUsed": "Yes" if fixed_pbn_palette_data is not None else "No"
                }
            )
            typer.echo(f"Final PBN quantized image saved to: {quantized_pbn_path} (using {len(final_pbn_palette)} colors).")
        else:
            typer.secho(f"Error: Quantization failed to return image or palette for {path_to_main_image_for_pbn_quantization}.", fg=typer.colors.RED); raise typer.Exit(code=1)

    except Exception as e: 
        typer.secho(f"Error during PBN quantization of {path_to_main_image_for_pbn_quantization}: {e}", fg=typer.colors.RED); traceback.print_exc(); raise typer.Exit(code=1)

    if final_pbn_palette is None: # Should not happen if previous try/except is robust
        typer.secho("Critical error: PBN palette not generated.", fg=typer.colors.RED); raise typer.Exit(code=1)

    canvas_size_for_final_output: Tuple[int, int]
    img_data_for_segmentation_shape: Tuple[int, int] # (height, width)
    try:
        # quantized_pbn_path is output_paths[PBNFile.QUANTIZED_GUIDE]
        with Image.open(quantized_pbn_path) as img_pil_for_segmentation:
            img_data_for_segmentation_shape = img_pil_for_segmentation.size[::-1] # height, width
            canvas_size_for_final_output = img_pil_for_segmentation.size # width, height
        
        if output_canvas_dimensions_px and output_canvas_dimensions_px != canvas_size_for_final_output:
             typer.secho(f"Internal Warning: Expected canvas dimensions {output_canvas_dimensions_px} but "
                        f"PBN quantized image is {canvas_size_for_final_output}. Using actual PBN image size for output.", fg=typer.colors.YELLOW)
        typer.echo(f"Using PBN image size for segmentation/output: {canvas_size_for_final_output[0]}x{canvas_size_for_final_output[1]} pixels (from '{quantized_pbn_path.name}').")
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
    
    if effective_font_size is None: # Should have a default from earlier
        effective_font_size = 10 

    if effective_min_scaling_font_size > effective_font_size:
        if enable_stable_font_scaling_cli:
            typer.secho(f"Warning: --min-scaling-font-size ({effective_min_scaling_font_size}) is greater than target font-size ({effective_font_size}). "
                        f"Adjusting minimum scaling font size to {effective_font_size}.", fg=typer.colors.YELLOW)
        effective_min_scaling_font_size = effective_font_size
    effective_min_scaling_font_size = max(1, effective_min_scaling_font_size)

    segment_call_kwargs = {
        "input_path": str(quantized_pbn_path), # Path to PBNFile.QUANTIZED_GUIDE
        "palette": final_pbn_palette, # palette from quantize_image
        "font_size": effective_font_size,
        "font_path": str(font_path) if font_path else None,
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
        if effective_dpi_val is None: # Should have a default
            effective_dpi_val = 96
            typer.secho("Warning: DPI for blobbify not determined, defaulting to 96.", fg=typer.colors.YELLOW)

        px_per_mm = effective_dpi_val / 25.4
        area_min_px = int(blob_min * (px_per_mm**2))
        area_max_px = int(blob_max * (px_per_mm**2))
        typer.echo(f"Blobbify: DPI={effective_dpi_val}, min_area={area_min_px}px² ({blob_min}mm²), max_area={area_max_px}px² ({blob_max}mm²).")
        # Ensure img_data_for_segmentation_shape is (height, width) as expected by blobbify_primitives
        primitives = segment.blobbify_primitives(primitives, img_data_for_segmentation_shape, area_min_px, area_max_px, min_label_font, interpolate_contours) # type: ignore
        typer.echo(f"After blobbification: {len(primitives)} regions processed.")

    typer.echo(f"Target SVG label color: {label_color}")
    typer.echo(f"Target outline color: {outline_color_cli}") 

    font_path_for_vector_str = str(font_path) if font_path else None
    if not raster_only and vector_path: # vector_path is output_paths[PBNFile.VECTOR_OUTPUT]
        try:
            # Prepare metadata for SVG
            svg_metadata = {
                "PbNgen-FileType": "Vector PBN Output", # Or PBNFile.VECTOR_OUTPUT.value
                "SourceQuantizedGuide": str(quantized_pbn_path.name),
                "PaletteColors": str(len(final_pbn_palette)),
                "LabelStrategy": effective_label_strategy or "default",
                "SmallRegionStrategy": small_region_label_strategy,
                "FontSize": str(effective_font_size)
            }
            file_utils.save_pbn_svg(
                vector_path, # This is a Path object
                canvas_size_for_final_output,
                primitives,
                font_path_str=font_path_for_vector_str,
                default_font_size=effective_font_size,
                label_color_str=label_color,
                outline_color_hex=outline_color_cli,
                additional_nudge_pixels_up=render_nudge_pixels_up,
                command_line_invocation=command_line_str,
                additional_metadata=svg_metadata
            )
            typer.echo(f"SVG output saved to: {vector_path}")
        except Exception as e:
            typer.secho(f"Error writing SVG output: {e}", fg=typer.colors.RED); traceback.print_exc()    

    try:
        # labeled_path is output_paths[PBNFile.RASTER_OUTPUT]
        labeled_img = segment.render_raster_from_primitives( # type: ignore
            canvas_size_for_final_output,
            primitives,
            font_path, # Path object or None 
            additional_nudge_pixels_up=render_nudge_pixels_up,
            label_text_color=label_color,
            outline_color_str_hex=outline_color_cli
        )
        file_utils.save_pbn_png(
            labeled_img,
            labeled_path, # Path object
            command_line_invocation=command_line_str,
            additional_metadata={
                "PbNgen-FileType": "Labeled Raster PBN Output", # Or PBNFile.RASTER_OUTPUT.value
                "SourceQuantizedGuide": str(quantized_pbn_path.name), # Use .name for relative path in metadata
                "PaletteColors": str(len(final_pbn_palette)),
                "LabelStrategy": effective_label_strategy or "default",
                "SmallRegionStrategy": small_region_label_strategy,
                "FontSize": str(effective_font_size)
            }
        )
        typer.echo(f"Labeled raster PNG output saved to: {labeled_path}")
    except Exception as e:
        typer.secho(f"Error rendering raster output: {e}", fg=typer.colors.RED)
        traceback.print_exc()
        raise typer.Exit(code=1)

    if not skip_legend:
        # legend_path is output_paths[PBNFile.PALETTE_LEGEND]
        try:
            legend_pil_image = legend.create_legend_image(
                final_pbn_palette, # palette from quantize_image
                font_path=str(font_path) if font_path else None,
                font_size=effective_font_size,
                swatch_size=swatch_size,
                padding=10 
            )

            if legend_pil_image:
                additional_legend_metadata = {
                    "PbNgen-FileType": "Palette Legend", # Or PBNFile.PALETTE_LEGEND.value
                    "PaletteColors": str(len(final_pbn_palette)),
                    "SwatchSize": str(swatch_size),
                    "LegendFontSize": str(effective_font_size)
                }
                file_utils.save_pbn_png(
                    legend_pil_image,
                    legend_path, # Path object
                    command_line_invocation=command_line_str,
                    additional_metadata=additional_legend_metadata
                )
                typer.echo(f"Palette legend saved to: {legend_path}")
            else:
                typer.secho(f"Warning: Palette legend image could not be generated (e.g., empty palette).", fg=typer.colors.YELLOW)

        except Exception as e:
            typer.secho(f"Error generating or saving palette legend: {e}", fg=typer.colors.RED)
            traceback.print_exc()    
    
    typer.secho("\nProcessing complete!", fg=typer.colors.GREEN)

    # UPDATED: --no-cruft logic is now safe, robust, and automatic.
    if no_cruft:
        typer.echo("\n--no-cruft active: Cleaning up intermediate files...")

        # Get the final state of the directory
        final_files_in_dir = set(os.listdir(output_dir))

        # Determine which files were created during this run
        created_files = final_files_in_dir - initial_files_in_dir

        # Define which files are considered final outputs and should NOT be deleted
        final_output_basenames = {
            PBN_FILE_BASENAMES[key] for key in FINAL_OUTPUT_ENUM_KEYS if key in PBN_FILE_BASENAMES
        }

        # Determine which of the created files are not final outputs
        files_to_delete = created_files - final_output_basenames
        
        if not files_to_delete:
            typer.echo("  No intermediate byproduct files were found to delete.")
        else:
            for filename in sorted(list(files_to_delete)): # Sort for deterministic output
                file_path_to_delete = output_dir / filename
                if file_path_to_delete.exists():
                    try:
                        file_path_to_delete.unlink()
                        typer.echo(f"  Deleted: {file_path_to_delete.name}")
                    except OSError as e:
                        typer.secho(f"  Error deleting {file_path_to_delete.name}: {e}", fg=typer.colors.RED)

    if output_dir: typer.echo(f"Outputs in: {output_dir.resolve()}")

if __name__ == "__main__":
    rich.traceback.install(show_locals=False, suppress=[typer, __name__]) # type: ignore
    typer.run(pbn_cli)