import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from PIL import Image, ImageDraw, ImageFont
import os

def segment_and_label(input_path, palette, font_path=None, font_size=None, tile_spacing=None, min_region_area=50):
    """
    Segment the image into regions by color and overlay tiled palette index labels
    on a white background with light blue outlines, with adaptive sizing.

    Args:
        input_path (str): Path to the quantized image.
        palette (np.ndarray): The palette used for quantization.
        font_path (str, optional): Path to a TTF font file.
        font_size (int, optional): Base font size; adaptive if None.
        tile_spacing (int, optional): Base spacing; adaptive if None.
        min_region_area (int): Minimum area of a region to annotate.

    Returns:
        Image.Image: The annotated image.
    """
    image = Image.open(input_path).convert("RGB")
    img_data = np.array(image)

    output = Image.new("RGB", image.size, color=(255, 255, 255))
    draw = ImageDraw.Draw(output)

    for idx, color in enumerate(palette):
        mask = np.all(img_data == color, axis=-1).astype(np.uint8)
        labeled_array, _ = label(mask)

        for region in regionprops(labeled_array):
            if region.area < min_region_area:
                continue

            minr, minc, maxr, maxc = region.bbox
            region_width = maxc - minc
            region_height = maxr - minr

            # Dynamically determine tile spacing and font size
            local_spacing = tile_spacing or max(8, min(region_width, region_height) // 4)
            local_font_size = font_size or int(local_spacing * 0.7)
            local_font_size = max(8, min(local_font_size, 24))

            if font_path and os.path.isfile(font_path):
                region_font = ImageFont.truetype(font_path, local_font_size)
            else:
                region_font = ImageFont.load_default()

            # Outline using find_boundaries for better visibility
            region_mask = (labeled_array == region.label).astype(np.uint8)
            boundaries = find_boundaries(region_mask, mode='outer')
            by, bx = np.where(boundaries)
            for y, x in zip(by, bx):
                draw.line([(x + minc, y + minr), (x + minc + 1, y + minr)], fill=(173, 216, 230))

            # Fill with multiple numbers per tile
            for y in range(int(minr), int(maxr), local_spacing):
                for x in range(int(minc), int(maxc), local_spacing):
                    if labeled_array[y, x] == region.label:
                        draw.text((x, y), str(idx), fill=(173, 216, 230), font=region_font)
                        draw.text((x + local_spacing // 2, y + local_spacing // 2), str(idx), fill=(173, 216, 230), font=region_font)

    return output
