import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from PIL import Image, ImageDraw, ImageFont
import os

from skimage.measure import find_contours
def collect_region_primitives(input_path, palette, font_path=None, font_size=None, tile_spacing=None, min_region_area=50):
    """
    Collect drawable region outlines and label positions for vector output.

    Args:
        input_path (str): Path to the quantized image.
        palette (np.ndarray): The palette used for quantization.
        font_size (int, optional): Base font size; adaptive if None.
        tile_spacing (int, optional): Base spacing; adaptive if None.
        min_region_area (int): Minimum area of a region to annotate.

    Returns:
        List[dict]: List of drawing primitives (outlines + text labels).
    """
    image = Image.open(input_path).convert("RGB")
    img_data = np.array(image)
    img_h, img_w = img_data.shape[:2]
    primitives = []

    for idx, color in enumerate(palette):
        mask = np.all(img_data == color, axis=-1).astype(np.uint8)
        labeled_array, _ = label(mask)
        
        for region in regionprops(labeled_array):
            if region.area < min_region_area:
                continue
        
            region_mask = (labeled_array == region.label).astype(np.uint8)
            contours = find_contours(region_mask, level=0.5)
            if not contours:
                continue
        
            longest = max(contours, key=len)
            outline = [(int(x), int(y)) for y, x in longest]
        
            minr, minc, maxr, maxc = region.bbox
            region_width = maxc - minc
            region_height = maxr - minr
            region_diag = (region_width**2 + region_height**2) ** 0.5
        
        #    local_spacing = tile_spacing or max(8, min(region_width, region_height) // 4)
        #    local_font_size = font_size or int(local_spacing * 0.7)
        #    local_font_size = max(8, min(local_font_size, 24))
            local_spacing = tile_spacing or max(6, int(region_diag / 8))
            local_font_size = font_size or int(local_spacing * 0.66)
            local_font_size = max(6, min(local_font_size, 48))
        
            # Load font
            if font_path and os.path.isfile(font_path):
                font = ImageFont.truetype(font_path, local_font_size)
            else:
                font = ImageFont.load_default()
        
            labels = []
            for y in range(int(minr), int(maxr), local_spacing):
                for x in range(int(minc), int(maxc), local_spacing):
                    if labeled_array[y, x] == region.label:
                        text = str(idx)
#                        text_width, text_height = font.getsize(text)

                        # create a dummy draw context
                        dummy_img = Image.new("RGB", (1, 1))
                        draw = ImageDraw.Draw(dummy_img)
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        # Check if the entire text box fits within the region
                        fits = True
                        for dy in range(text_height):
                            for dx in range(text_width):
                                px = x + dx
                                py = y + dy
                                if px >= labeled_array.shape[1] or py >= labeled_array.shape[0] or labeled_array[py, px] != region.label:
                                    fits = False
                                    break
                            if not fits:
                                break
                        if fits:
                            labels.append({
                                "position": (x, y),
                                "value": text,
                                "font_size": local_font_size
                            })
                            labels.append({
                                "position": (x + local_spacing // 2, y + local_spacing // 2),
                                "value": text,
                                "font_size": local_font_size
                            })
        
            primitives.append({
                "outline": outline,
                "labels": labels,
                "region_id": idx,
                "color": tuple(color.tolist())
            })
        
    return primitives


def segment_and_label(input_path, palette, font_path=None, font_size=None, tile_spacing=None, min_region_area=50):
    """
    Raster-based fallback rendering of region primitives to a Pillow image.
    """
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    output = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output)

    primitives = collect_region_primitives(
        input_path=input_path,
        palette=palette,
        font_size=font_size,
        tile_spacing=tile_spacing,
        min_region_area=min_region_area
    )

    for region in primitives:
        for x, y in region["outline"]:
            draw.point((x, y), fill=region["color"])

        for label in region["labels"]:
            if font_path and os.path.isfile(font_path):
                font = ImageFont.truetype(font_path, label["font_size"])
            else:
                font = ImageFont.load_default()
            draw.text(label["position"], label["value"], fill=region["color"], font=font)

    return output
