import sys
from pathlib import Path
from PIL import Image, PngImagePlugin
from typing import Optional, Dict, Tuple, List # Added Tuple, List for SVG

# Imports for SVG functionality
import svgwrite
import base64
from xml.etree.ElementTree import Element, SubElement
import re

default_file_names = {
    "bpp_quantized_input": "input-bpp_quantized.png",
    "filtered_input": "input-filtered.png",
    "canvas_scaled_input": "input-canvas_scaled.png",
    "bpp_quantized_palette_input": "input_palette-bpp_quantized.png",
    "quantized_guide": "pbn_guide-ncolor_quantized.png",
    "vector_output": "vector-pbn_canvas.svg", # Already aware of vector output
    "raster_output": "raster-pbn_canvas.png",
    "palette_legend": "palette-pbn_legend.png",
}

def save_pbn_png(
    image_to_save: Image.Image,
    output_path: Path,
    command_line_invocation: Optional[str] = None,
    additional_metadata: Optional[Dict[str, str]] = None
):
    """
    Saves a PIL Image object as a PNG file, embedding specified metadata.
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    png_info = PngImagePlugin.PngInfo()

    if command_line_invocation:
        png_info.add_text("pbngen:command_line", command_line_invocation)

    if additional_metadata:
        for key, value in additional_metadata.items():
            # Sanitize key slightly for consistency if needed, though PngInfo is flexible
            key_clean = key.replace(" ", "_")
            png_info.add_text(f"pbngen:{key_clean}", str(value))

    try:
        image_to_save.save(output_path, "PNG", pnginfo=png_info)
    except Exception as e:
        print(f"Error saving PNG {output_path}: {e}")

def save_pbn_svg(
    output_path: Path,
    canvas_size: Tuple[int, int],
    primitives: List[dict],
    font_path_str: Optional[str] = None,
    default_font_size: Optional[int] = 10,
    label_color_str: Optional[str] = "#88ddff",
    outline_color_hex: str = "#88ddff",
    command_line_invocation: Optional[str] = None,
    additional_metadata: Optional[dict[str, str]] = None
):
    """
    Render collected drawing primitives into an SVG file with embedded metadata.
    """
    width, height = canvas_size

    pbngen_ns_uri = 'http://www.github.com/scottvr/pbngen/pbngen-ns#'
    pbngen_ns_prefix = 'pbngen'

    # Initialize drawing WITHOUT the custom xmlns in the constructor's **extra
    dwg = svgwrite.Drawing(
        str(output_path),
        size=(f"{width}px", f"{height}px"),
        profile='full'
    )

    # Add the custom namespace declaration directly to the root svg element's attributes
    dwg.attribs[f'xmlns:{pbngen_ns_prefix}'] = pbngen_ns_uri

    # --- Font embedding logic
    font_family_svg = "sans-serif"
    if font_path_str:
        try:
            font_path_obj = Path(font_path_str)
            if font_path_obj.is_file() and font_path_obj.suffix.lower() in ['.ttf', '.otf', '.woff', '.woff2']:
                font_family_svg = font_path_obj.stem
                try:
                    with open(font_path_obj, 'rb') as f_font:
                        font_data = f_font.read()
                    font_data_b64 = base64.b64encode(font_data).decode('utf-8')
                    font_mime_type = "font/ttf"
                    if font_path_obj.suffix.lower() == ".otf": font_mime_type = "font/otf"
                    elif font_path_obj.suffix.lower() == ".woff": font_mime_type = "application/font-woff"
                    elif font_path_obj.suffix.lower() == ".woff2": font_mime_type = "font/woff2"
                    font_face_css = f"@font-face {{"
                    font_face_css += f"font-family: '{font_family_svg}';"
                    font_face_css += f"src: url(data:{font_mime_type};base64,{font_data_b64});"
                    font_face_css += f"}}"
                    dwg.defs.add(dwg.style(content=font_face_css))
                except Exception as e:
                    print(f"Warning: Could not embed font {font_path_str} into SVG: {e}")
        except Exception as e:
            print(f"Warning: Error processing font path '{font_path_str}' for SVG: {e}")
    # --- End Font embedding logic ---

    # --- Metadata block ---
    if command_line_invocation or additional_metadata:
        metadata_root_element = Element('metadata') # Standard SVG <metadata> tag
        metadata_root_element.set('id', 'pbngenApplicationMetadata')

        if command_line_invocation:
            # Use the prefixed name for the custom element
            cli_el = SubElement(metadata_root_element, f'{pbngen_ns_prefix}:CommandLineInvocation')
            cli_el.text = command_line_invocation
        
        if additional_metadata:
            for key, value in additional_metadata.items():
                el_name_local = key
                el_name_local = re.sub(r'[^\w.-]', '_', el_name_local) # Allow word chars, '.', '-'
                if not el_name_local or not (el_name_local[0].isalpha() or el_name_local[0] == '_'):
                    el_name_local = '_' + el_name_local
                
                # Use the prefixed name for the custom element
                qualified_name = f'{pbngen_ns_prefix}:{el_name_local}'
                meta_item_el = SubElement(metadata_root_element, qualified_name)
                meta_item_el.text = str(value)
        
        dwg.elements.append(metadata_root_element)
    # --- End Metadata block ---

    dwg.add(dwg.rect(insert=(0, 0), size=(f"{width}px", f"{height}px"), fill='white', stroke='gray', stroke_width=1))

    for item in primitives:
        for contour in item.get("outline", []):
            filtered_points = [(max(0, min(int(x), width)), max(0, min(int(y), height))) for x, y in contour]
            if len(filtered_points) > 1:
                dwg.add(dwg.polyline(points=filtered_points, stroke=outline_color_hex, fill="none", stroke_width=1))
        for label in item.get("labels", []):
            x, y = label["position"]
            font_size_label = label.get("font_size", default_font_size if default_font_size else 10)
            dwg.add(dwg.text(str(label["value"]), insert=(int(x), int(y)), fill=label_color_str,
                             font_family=font_family_svg, font_size=f"{font_size_label}px",
                             text_anchor="middle", alignment_baseline="middle"))
    
    dwg.save()