import svgwrite
from typing import Tuple, Optional, List # Added Optional, List
from pathlib import Path # For font handling
import base64 # For embedding font

def write_svg(
    output_path: str,
    canvas_size: Tuple[int, int],
    primitives: List[dict], # More specific type hint
    font_path_str: Optional[str] = None, # Added
    default_font_size: Optional[int] = 10 # Added, though label["font_size"] is primary
):
    """
    Render collected drawing primitives into an SVG file with optional debugging.
    Uses font information if provided.
    """
    width, height = canvas_size
    dwg = svgwrite.Drawing(output_path, size=(f"{width}px", f"{height}px"), profile='full') # Added units for clarity

    # Define default font family
    font_family_svg = "sans-serif" # Generic fallback

    if font_path_str:
        try:
            font_path_obj = Path(font_path_str)
            if font_path_obj.is_file() and font_path_obj.suffix.lower() in ['.ttf', '.otf', '.woff', '.woff2']:
                # Attempt to get font family name from font file (can be complex)
                # For simplicity, we'll use the filename as a stand-in or a predefined name
                # A more robust way would involve a font parsing library.
                font_family_svg = font_path_obj.stem # Use filename stem as font family name

                # Option 1: Embed font directly in SVG (increases file size, best portability)
                # This requires reading the font file and base64 encoding it.
                # Note: SVG viewers have varying support for embedded WOFF/WOFF2 vs TTF/OTF.
                # WOFF is generally preferred for web.
                try:
                    with open(font_path_obj, 'rb') as f_font:
                        font_data = f_font.read()
                    font_data_b64 = base64.b64encode(font_data).decode('utf-8')
                    
                    font_mime_type = "font/ttf" # Default
                    if font_path_obj.suffix.lower() == ".otf":
                        font_mime_type = "font/otf"
                    elif font_path_obj.suffix.lower() == ".woff":
                        font_mime_type = "application/font-woff"
                    elif font_path_obj.suffix.lower() == ".woff2":
                        font_mime_type = "font/woff2"

                    style_def = f"@font-face {{"
                    style_def += f"font-family: '{font_family_svg}';"
                    style_def += f"src: url(data:{font_mime_type};base64,{font_data_b64});"
                    style_def += f"}}"
                    dwg.defs.add(dwg.style(style_def))
                except Exception as e:
                    print(f"Warning: Could not embed font {font_path_str} into SVG: {e}")
                    # Fallback to using font_family_svg name, assuming it might be installed
                    pass # font_family_svg will still be font_path_obj.stem

        except Exception as e:
            print(f"Warning: Error processing font path '{font_path_str}' for SVG: {e}")
            # font_family_svg remains "sans-serif"

    # Optional debug box to visualize canvas bounds
    dwg.add(dwg.rect(insert=(0, 0), size=(f"{width}px", f"{height}px"), fill='white', stroke='gray', stroke_width=1))

    outline_color = "#88ddff"  
    label_color = "#88ddff"

    for item in primitives:
        # Draw outlines
        for contour in item.get("outline", []):
            # Ensure points are within canvas, primarily a safeguard
            filtered_points = []
            for x_coord, y_coord in contour:
                # Clamp coordinates to be within canvas, though ideally they should be.
                # SVG can handle points outside, but it's good practice.
                # For PBN, exact outline is important.
                clamped_x = max(0, min(int(x_coord), width))
                clamped_y = max(0, min(int(y_coord), height))
                filtered_points.append((clamped_x, clamped_y))
            
            if filtered_points and len(filtered_points) > 1: # Need at least 2 points for a polyline
                dwg.add(dwg.polyline(
                    points=filtered_points,
                    stroke=outline_color,
                    fill="none",
                    stroke_width=1
                ))

        # Draw text labels
        for label in item.get("labels", []):
            x, y = label["position"]
            # Ensure label position is within canvas to avoid potential rendering issues
            # Although SVG clips, this is a good check.
            if not (0 <= x < width and 0 <= y < height): 
                # Optionally, clamp label position or skip
                # For now, we'll keep original behavior of skipping if outside initial check
                # but vector_output's original code does not have this check, it's in segment.py
                # For safety, let's ensure it's drawn if segment.py allowed it.
                pass # Allow drawing even if slightly outside, SVG will clip.

            font_size_label = label.get("font_size", default_font_size if default_font_size else 10) #

            dwg.add(dwg.text(
                str(label["value"]), # Ensure value is string
                insert=(int(x), int(y)),
                fill=label_color,
                # stroke=label_color, # Usually not needed if fill is present, can make text look thicker
                font_family=font_family_svg, # Apply the determined font family
                font_size=f"{font_size_label}px", # Add units
                text_anchor="middle", #
                alignment_baseline="middle" #
            ))

    dwg.save()