from PIL import Image, ImageFilter
from typing import Optional

def apply_style(input_path, output_path, style, blur_radius: Optional[int] = None, pixelate_block_size: Optional[int] = None, mosaic_block_size: Optional[int] = None):
    """
    Apply a visual style to an input image and save it.

    Args:
        input_path (str): Path to the original image.
        output_path (str): Path to save the styled image.
        style (str): Name of the style to apply (e.g. blur, mosaic, pixelate).
    """
    image = Image.open(input_path).convert("RGB")

    if style == "blur":
        if not blur_radius:
            blur_radius = 4

        styled = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    elif style == "pixelate":
        # Resize down and back up to simulate pixelation
        w, h = image.size
        if not pixelate_block_size: # user passed no argument
            pixelate_block_size = 64
        pixel_size = max(4, min(w, h) // pixelate_block_size)
        styled = image.resize((w // pixel_size, h // pixel_size), resample=Image.NEAREST)
        styled = styled.resize((w, h), resample=Image.NEAREST)

    elif style == "mosaic":
        # Similar to pixelate but with bicubic upscale for blocky blending
        w, h = image.size
        if not mosaic_block_size:
            mosaic_block_size = 64
        block_size = max(4, min(w, h) // mosaic_block_size)
        styled = image.resize((w // block_size, h // block_size), resample=Image.NEAREST)
        styled = styled.resize((w, h), resample=Image.BICUBIC)

    else:
        raise ValueError(f"Unsupported style: {style}")

    styled.save(output_path)
