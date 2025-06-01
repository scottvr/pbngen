import sys
from pathlib import Path
from PIL import Image, PngImagePlugin 
from typing import Optional, Dict 

default_file_names = {
    "bpp_quantized_input": "input-bpp_quantized.png",
    "filtered_input": "input-filtered.png", 
    "canvas_scaled_input": "input-canvas_scaled.png", 
    "bpp_quantized_palette_input": "input_palette-bpp_quantized.png",
    "quantized_guide": "pbn_guide-ncolor_quantized.png", 
    "vector_output": "vector-pbn_canvas.svg", 
    "raster_output": "raster-pbn_canvas.png",
    "palette_legend": "palette-pbn_legend.png", 
}

def save_pbn_png(
    image_to_save: Image.Image,
    output_path: Path, # Direct path to save to
    command_line_invocation: Optional[str] = None,
    additional_metadata: Optional[Dict[str, str]] = None
):
    """
    Saves a PIL Image object as a PNG file, embedding specified metadata.

    Args:
        image_to_save: The PIL.Image.Image object to save.
        output_path: The pathlib.Path object for the output file.
        command_line_invocation: The string capturing the command line call.
        additional_metadata: A dictionary of other key-value pairs for metadata.
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    png_info = PngImagePlugin.PngInfo()

    if command_line_invocation:
        png_info.add_text("pbngen:command_line", command_line_invocation)
    
    # You could add other fixed metadata here too, e.g., software version
    # png_info.add_text("Software", "pbngen vX.Y.Z")

    if additional_metadata:
        for key, value in additional_metadata.items():
            png_info.add_text(key, str(value)) # Ensure value is string

    try:
        image_to_save.save(output_path, "PNG", pnginfo=png_info)
        # typer.echo(f"Saved PNG with metadata: {output_path.name}") # Optional: for verbose output
    except Exception as e:
        # Consider how to handle errors, e.g., log them or raise them
        # For CLI, typer.secho might be appropriate if this function is in pbngen.py
        print(f"Error saving PNG {output_path}: {e}") # Or use typer.secho