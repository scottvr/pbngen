from PIL import Image, ImageFilter


def apply_style(input_path, output_path, style):
    """
    Apply a visual style to an input image and save it.

    Args:
        input_path (str): Path to the original image.
        output_path (str): Path to save the styled image.
        style (str): Name of the style to apply (e.g. blur, mosaic, pixelate).
    """
    image = Image.open(input_path).convert("RGB")

    if style == "blur":
        styled = image.filter(ImageFilter.GaussianBlur(radius=4))

    elif style == "pixelate":
        # Resize down and back up to simulate pixelation
        w, h = image.size
        pixel_size = max(4, min(w, h) // 64)
        styled = image.resize((w // pixel_size, h // pixel_size), resample=Image.NEAREST)
        styled = styled.resize((w, h), resample=Image.NEAREST)

    elif style == "mosaic":
        # Similar to pixelate but with bicubic upscale for blocky blending
        w, h = image.size
        block_size = max(4, min(w, h) // 64)
        styled = image.resize((w // block_size, h // block_size), resample=Image.NEAREST)
        styled = styled.resize((w, h), resample=Image.BICUBIC)

    else:
        raise ValueError(f"Unsupported style: {style}")

    styled.save(output_path)
