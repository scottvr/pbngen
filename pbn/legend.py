from PIL import Image, ImageDraw, ImageFont
import os #

def create_legend_image(palette, font_path=None, font_size=14, swatch_size=40, padding=10):
    """
    Creates a palette legend PIL Image object.

    Args:
        palette (list or np.ndarray): Color palette, each item an RGB tuple/list or ndarray.
        font_path (str, optional): Path to a TTF font file.
        font_size (int): Font size for the index numbers.
        swatch_size (int): Width/height of each color swatch.
        padding (int): Space around elements and between swatches.

    Returns:
        PIL.Image.Image: The generated legend image, or None if num_colors is 0.
    """
    num_colors = len(palette) #
    if num_colors == 0:
        return None # Or raise an error, or return a minimal placeholder

    # Calculate image dimensions
    width = (swatch_size * num_colors) + (padding * (num_colors + 1)) #
    height = swatch_size + (2 * padding) #

    image = Image.new("RGB", (width, height), color=(255, 255, 255)) #
    draw = ImageDraw.Draw(image) #

    loaded_font = None
    try:
        if font_path and os.path.isfile(font_path): #
            loaded_font = ImageFont.truetype(font_path, font_size) #
    except IOError:
        pass # Will fall through to default if custom font fails

    if not loaded_font: # If custom font path not provided, invalid, or failed to load
        try:
            # Try to load default font with specified size
            loaded_font = ImageFont.load_default(size=font_size)
        except TypeError: # Older Pillow versions might not support size for load_default
            loaded_font = ImageFont.load_default() # (adapted)


    for idx, color_data in enumerate(palette): #
        x_start_swatch = padding + idx * (swatch_size + padding) #
        y_start_swatch = padding #

        fill_color = (0,0,0) # Default to black if color conversion fails
        if hasattr(color_data, 'tolist'): # Handles numpy array elements
            fill_color = tuple(int(c) for c in color_data.tolist()) #
        elif isinstance(color_data, (list, tuple)) and len(color_data) == 3:
            fill_color = tuple(int(c) for c in color_data)
        # else: color_data might be in an unexpected format, using default black or could raise error

        draw.rectangle(
            [x_start_swatch, y_start_swatch, x_start_swatch + swatch_size, y_start_swatch + swatch_size], #
            fill=fill_color, #
            outline=(0, 0, 0) #
        )

        text_content = str(idx) #
        
        # Text placement logic using textbbox for better centering
        # (0,0) is a reference point for textbbox, not the final drawing position.
        try: # Modern Pillow (9.2.0+)
            bbox = loaded_font.getbbox(text_content) # Returns (x1, y1, x2, y2)
            text_draw_x_offset = bbox[0]
            text_draw_y_offset = bbox[1]
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError: # Fallback for older Pillow versions
            # Using draw.textbbox (less ideal than font.getbbox but better than nothing)
            bbox_calc = draw.textbbox((0,0), text_content, font=loaded_font) #
            text_draw_x_offset = bbox_calc[0]
            text_draw_y_offset = bbox_calc[1]
            text_w = bbox_calc[2] - bbox_calc[0] #
            text_h = bbox_calc[3] - bbox_calc[1] #

        # Center text within the swatch
        # Adjust by the text_draw_x/y_offset to account for the glyph's actual position relative to its origin
        text_x_position = x_start_swatch + (swatch_size - text_w) / 2.0 - text_draw_x_offset
        text_y_position = y_start_swatch + (swatch_size - text_h) / 2.0 - text_draw_y_offset
        
        draw.text((text_x_position, text_y_position), text_content, fill=(0, 0, 0), font=loaded_font) #

    return image