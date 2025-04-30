from PIL import Image, ImageDraw, ImageFont
import os

def generate_legend(palette, output_path, font_path=None, font_size=14, swatch_size=40, padding=10):
    """
    Create a palette legend image.

    Args:
        palette (np.ndarray): Color palette as an array of RGB values.
        output_path (str): Where to save the resulting legend image.
        font_path (str, optional): Path to a TTF font file.
        font_size (int): Font size for the index numbers.
        swatch_size (int): Width/height of each color swatch.
        padding (int): Space between swatches and text.
    """
    num_colors = len(palette)
    width = swatch_size * num_colors + padding * (num_colors + 1)
    height = swatch_size + 2 * padding

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    if font_path and os.path.isfile(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    for idx, color in enumerate(palette):
        x = padding + idx * (swatch_size + padding)
        y = padding
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(color.tolist()), outline=(0, 0, 0))
        text = str(idx)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = x + (swatch_size - text_w) // 2
        text_y = y + (swatch_size - text_h) // 2
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

    image.save(output_path)
