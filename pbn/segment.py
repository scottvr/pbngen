import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from PIL import Image, ImageDraw, ImageFont
import os


def find_stable_label_pixel(region_mask):
    """Find the most surrounded pixel in a binary region mask."""
    h, w = region_mask.shape
    best_score = -1
    best_pixel = (0, 0)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if region_mask[y, x] == 0:
                continue
            up = down = left = right = 0
            while y - up - 1 >= 0 and region_mask[y - up - 1, x]: up += 1
            while y + down + 1 < h and region_mask[y + down + 1, x]: down += 1
            while x - left - 1 >= 0 and region_mask[y, x - left - 1]: left += 1
            while x + right + 1 < w and region_mask[y, x + right + 1]: right += 1
            score = up * down * left * right
            if score > best_score:
                best_score = score
                best_pixel = (x, y)
    return best_pixel


def collect_region_primitives(input_path, palette, font_size=None, tile_spacing=None,
                              min_region_area=50, label_mode="diagonal"):
    """
    Collect drawable region outlines and label positions for vector output.
    """
    image = Image.open(input_path).convert("RGB")
    img_data = np.array(image)
    primitives = []

    for idx, color in enumerate(palette):
        mask = np.all(img_data == color, axis=-1).astype(np.uint8)
        labeled_array, _ = label(mask)

        for region in regionprops(labeled_array):
            if region.area < min_region_area:
                continue

            region_mask = (labeled_array == region.label).astype(np.uint8)
            boundaries = find_boundaries(region_mask, mode='outer')
            by, bx = np.where(boundaries)
            outline = [(int(x), int(y)) for y, x in zip(by, bx)]

            minr, minc, maxr, maxc = region.bbox
            region_width = maxc - minc
            region_height = maxr - minr

            local_spacing = tile_spacing or max(8, min(region_width, region_height) // 4)
            local_font_size = font_size or int(local_spacing * 0.7)
            local_font_size = max(8, min(local_font_size, 24))

            labels = []

            if label_mode == "centroid":
                cx, cy = region.centroid
                labels = [{
                    "position": (int(cx), int(cy)),
                    "value": str(idx),
                    "font_size": local_font_size
                }]

            elif label_mode == "stable":
                sx, sy = find_stable_label_pixel(region_mask)
                labels = [{
                    "position": (sx + minc, sy + minr),
                    "value": str(idx),
                    "font_size": local_font_size
                }]

            elif label_mode == "diagonal":
                for y in range(int(minr), int(maxr), local_spacing):
                    for x in range(int(minc), int(maxc), local_spacing):
                        if labeled_array[y, x] == region.label:
                            labels.append({
                                "position": (x, y),
                                "value": str(idx),
                                "font_size": local_font_size
                            })
                            labels.append({
                                "position": (x + local_spacing // 2, y + local_spacing // 2),
                                "value": str(idx),
                                "font_size": local_font_size
                            })

            else:
                raise ValueError(f"Unknown label_mode: {label_mode}")

            primitives.append({
                "outline": outline,
                "labels": labels,
                "region_id": idx,
                "color": (173, 216, 230)
            })

    return primitives


def segment_and_label(input_path, palette, font_path=None, font_size=None,
                      tile_spacing=None, min_region_area=50, label_mode="diagonal"):
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
        min_region_area=min_region_area,
        label_mode=label_mode
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
