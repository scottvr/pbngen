from skimage.measure import regionprops, find_contours
from scipy.ndimage import label
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def make_label(x, y, value, font_size):
    label = {
        "position": (x, y),
        "value": str(value),
        "font_size": font_size
    }
    return label

def find_stable_label_pixel(region_mask):
    """
    Find the most 'stable' label location within a binary region.
    A stable location is one that is most deeply surrounded by the same value
    in all four cardinal directions. Hat tip to Dan Munro for his pbnify JS that inspired this.

    Args:
        region_mask (np.ndarray): Binary 2D array where 1=region, 0=outside

    Returns:
        (int, int): (x, y) coordinate of the most stable pixel
    """
    h, w = region_mask.shape
    best_score = -1
    best_coord = (0, 0)

    def same_count(x, y, dx, dy):
        count = -1
        while 0 <= x < w and 0 <= y < h and region_mask[y, x] == 1:
            count += 1
            x += dx
            y += dy
        return count

    # Get all region pixel coords
    ys, xs = np.nonzero(region_mask)
    for x, y in zip(xs, ys):
        score = (
            same_count(x, y, -1, 0) *  # left
            same_count(x, y, 1, 0) *   # right
            same_count(x, y, 0, -1) *  # up
            same_count(x, y, 0, 1)     # down
        )
        if score > best_score:
            best_score = score
            best_coord = (x, y)

    return best_coord


def collect_region_primitives(input_path, palette, font_size=None, font_path=None, tile_spacing=None,
                              min_region_area=50, label_mode="diagonal"):
    """
    Collect drawable region outlines and label positions for vector output.
    - Uses ordered contours for outline
    - Skips labeling regions that span more than 95% of the image (likely background)
    - Assigns a unique region_id per distinct region
    """
    image = Image.open(input_path).convert("RGB")
    img_data = np.array(image)
    height, width = img_data.shape[:2]
    img_area = height * width
    primitives = []
    region_id_counter = 0

    for idx, color in enumerate(palette):
        mask = np.all(img_data == color, axis=-1).astype(np.uint8)
        labeled_array, _ = label(mask)

        for region in regionprops(labeled_array):
            if region.area < min_region_area:
                continue
            if region.area > 0.95 * img_area:
                continue  # likely background

            region_mask = (labeled_array == region.label).astype(np.uint8)
            contours = find_contours(region_mask, level=0.5)
            if not contours:
                continue

#            # First contour only
#            contour = contours[0]
            minr, minc, maxr, maxc = region.bbox
#            outline = []
#            for c in contour:
#                x = int(np.floor(c[1]))
#                y = int(np.floor(c[0]))
#                if 0 <= x < width and 0 <= y < height:
#                    outline.append((x, y))
            outlines = []

            for contour in contours:
                dense = interpolate_contour(contour, step=0.5)
                outline = []
                for x_f, y_f in dense:
                    x = int(np.floor(x_f))
                    y = int(np.floor(y_f))
                    if 0 <= x < width and 0 <= y < height:
                        outline.append((x, y))
                if outline:
                    outlines.append(outline)

            region_width = maxc - minc
            region_height = maxr - minr
            local_spacing = tile_spacing or max(8, min(region_width, region_height) // 4)
            local_font_size = font_size or int(local_spacing * 0.7)
            local_font_size = max(8, min(local_font_size, 24))

            labels = []

            if label_mode == "centroid":
                cx, cy = region.centroid
                cx = int(cx)
                cy = int(cy)
                print(f"label_pixel = ({cx}, {cy})")
            
                if 0 <= cx < width and 0 <= cy < height:
                    labels = [make_label(cx, cy, idx, local_font_size)]

            elif label_mode == "stable":
                sx, sy = find_stable_label_pixel(region_mask)
                print(f"label_pixel = ({sx}, {sy})")
                x = sx + minc
                y = sy + minr
                if 0 <= x < width and 0 <= y < height:
                    labels = [make_label(x, y, idx, local_font_size)]

            elif label_mode == "diagonal":
                for y in range(minr, min(maxr, height), local_spacing):
                    for x in range(minc, min(maxc, width), local_spacing):
                        if labeled_array[y, x] == region.label:
                            if x < width and y < height:
                                labels.append(make_label(x, y, idx, local_font_size))
                            if x + local_spacing // 2 < width and y + local_spacing // 2 < height:
                                labels.append(make_label(x, y, idx, local_font_size))
            else:
                raise ValueError(f"Unknown label_mode: {label_mode}")

            primitives.append(dict(
                outline=outlines,
                labels=labels,
                region_id=region_id_counter,
                color=tuple(int(c) for c in color)
            ))
            region_id_counter += 1

    return primitives

def segment_and_label(input_path, palette, font_path=None, font_size=None,
                      tile_spacing=None, min_region_area=50, label_mode="diagonal"):
    """
    Raster-based fallback rendering of region primitives to a Pillow image.

    Args:
        input_path (str): Path to the quantized input image.
        palette (np.ndarray): Color palette used for segmentation.
        font_path (str, optional): Path to TTF font.
        font_size (int, optional): Base font size.
        tile_spacing (int, optional): Diagonal label spacing.
        min_region_area (int): Minimum region size to keep.
        label_mode (str): Label strategy ("diagonal", "centroid", "stable").

    Returns:
        PIL.Image: Labeled image with outlines and text.
    """
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    output = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output)

    primitives = collect_region_primitives(
        input_path=input_path,
        palette=palette,
        font_size=font_size,
        font_path=font_path,
        tile_spacing=tile_spacing,
        min_region_area=min_region_area,
        label_mode=label_mode
    )

    for region in primitives:
        #color = region["color"]
        color = (102, 204, 255)

        for x, y in region["outline"]:
            draw.point((x, y), fill=color)

        for label in region["labels"]:
            if font_path and os.path.isfile(font_path):
                font = ImageFont.truetype(font_path, label["font_size"])
            else:
                font = ImageFont.load_default()

            draw.text(label["position"], label["value"], fill=color, font=font)

    return output

def interpolate_contour(contour, step=0.5):
    """Given a list of float (y, x) contour points, return a densified list of (x, y) points."""
    dense_points = []
    for i in range(len(contour) - 1):
        y0, x0 = contour[i]
        y1, x1 = contour[i + 1]
        dist = np.hypot(x1 - x0, y1 - y0)
        n_steps = max(1, int(dist / step))
        for j in range(n_steps):
            t = j / n_steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            dense_points.append((x, y))
    return dense_points

def render_raster_from_primitives(canvas_size, primitives, font_path=None):
    """
    Render outlines and labels from previously collected primitives to a Pillow image.
    All lines and labels use light blue for PBN clarity.
    """
    width, height = canvas_size
    output = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output)
    pbn_color = (102, 204, 255)

    for region in primitives:
        # Draw outlines
        for contour in region["outline"]:
            for x, y in contour:
                if 0 <= x < width and 0 <= y < height:
                    draw.point((x, y), fill=pbn_color)

        # Draw labels
        for label in region["labels"]:
            font_size = label["font_size"]
            if font_path and os.path.isfile(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()

            draw.text(label["position"], label["value"], fill=pbn_color, font=font)

    return output
