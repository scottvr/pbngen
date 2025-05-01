from skimage.measure import regionprops, find_contours
from scipy.ndimage import label as nd_label
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random

def make_label(x, y, value, font_size):
    return {"position": (x, y), "value": str(value), "font_size": font_size}

def find_stable_label_pixel(region_mask):
    h, w = region_mask.shape
    best_score, best_coord = -1, (0, 0)

    def same_count(x, y, dx, dy):
        count = -1
        while 0 <= x < w and 0 <= y < h and region_mask[y, x]:
            count += 1
            x += dx
            y += dy
        return count

    ys, xs = np.nonzero(region_mask)
    for x, y in zip(xs, ys):
        score = (same_count(x, y, -1, 0) * same_count(x, y, 1, 0) *
                 same_count(x, y, 0, -1) * same_count(x, y, 0, 1))
        if score > best_score:
            best_score = score
            best_coord = (x, y)

    return best_coord

def interpolate_contour(contour, step=0.5):
    dense = []
    for i in range(len(contour) - 1):
        y0, x0 = contour[i]
        y1, x1 = contour[i + 1]
        dist = np.hypot(x1 - x0, y1 - y0)
        for j in range(max(1, int(dist / step))):
            t = j / max(1, int(dist / step))
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            dense.append((x, y))
    return dense

def render_raster_from_primitives(canvas_size, primitives, font_path=None):
    width, height = canvas_size
    output = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output)
    pbn_color = (102, 204, 255)

    for region in primitives:
        for contour in region["outline"]:
            for x, y in contour:
                if 0 <= x < width and 0 <= y < height:
                    draw.point((x, y), fill=pbn_color)

        for label in region["labels"]:
            font_size = label["font_size"]
            if font_path and os.path.isfile(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
            draw.text(label["position"], label["value"], fill=pbn_color, font=font)

    return output

def blobbify_region(region_mask, min_blob_area, max_blob_area):
    h, w = region_mask.shape
    total_area = np.sum(region_mask)
    target_blob_area = (min_blob_area + max_blob_area) // 2
    num_blobs = max(1, total_area // target_blob_area)

    ys, xs = np.nonzero(region_mask)
    if len(xs) < num_blobs:
        return [region_mask]

    seed_coords = random.sample(list(zip(xs, ys)), num_blobs)
    distance_field = np.full_like(region_mask, np.inf, dtype=np.float32)
    label_field = np.zeros_like(region_mask, dtype=np.int32)

    for i, (sx, sy) in enumerate(seed_coords, start=1):
        dist = np.sqrt((np.arange(h)[:, None] - sy)**2 + (np.arange(w)[None, :] - sx)**2)
        mask = (dist < distance_field) & (region_mask == 1)
        label_field[mask] = i
        distance_field[mask] = dist[mask]

    blobs = []
    for i in range(1, num_blobs + 1):
        blob = (label_field == i).astype(np.uint8)
        area = blob.sum()
        if min_blob_area <= area <= max_blob_area:
            blobs.append(blob)

    return blobs

def blobbify_primitives(primitives, img_shape, min_blob_area, max_blob_area, min_label_font_size=8):
    h, w = img_shape
    new_primitives = []
    region_id_counter = 0

    for region in primitives:
        color = region["color"]
        outlines = region.get("outline", [])
        if not outlines:
            continue

        from PIL import Image as PILImage
        mask_img = PILImage.new("1", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        for contour in outlines:
            draw.polygon(contour, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8)

        blobs = blobbify_region(mask, min_blob_area, max_blob_area)
        for blob_mask in blobs:
            sub_labeled, _ = nd_label(blob_mask)
            for subregion in regionprops(sub_labeled):
                if subregion.area < min_blob_area:
                    continue
                sub_mask = (sub_labeled == subregion.label).astype(np.uint8)
                contours = find_contours(sub_mask, level=0.5)
                if not contours:
                    continue
                dense_outlines = []
                for contour in contours:
                    dense = interpolate_contour(contour, step=0.5)
                    outline = [(int(xf), int(yf)) for yf, xf in dense if 0 <= int(xf) < w and 0 <= int(yf) < h]
                    if outline:
                        dense_outlines.append(outline)
                sx, sy = find_stable_label_pixel(sub_mask)
                minr, minc, _, _ = subregion.bbox
                x, y = sx + minc, sy + minr
                label_font_size = max(min_label_font_size, min(24, int(np.sqrt(subregion.area) / 3)))
                if label_font_size < min_label_font_size:
                    continue
                original_palette_index = region["palette_index"] 
                label = make_label(x, y, region.get("label", "?"), label_font_size)
                new_primitives.append(dict(
                    outline=dense_outlines,
                    labels=[label],
                    region_id=f'blob_{region_id_counter}',
                    color=color,
                    palette_index=original_palette_index,
                ))
                region_id_counter += 1

    return new_primitives

def collect_region_primitives(input_path, palette, font_size=None, font_path=None, tile_spacing=None,
                              min_region_area=50, label_mode="diagonal"):
    image = Image.open(input_path).convert("RGB")
    img_data = np.array(image)
    height, width = img_data.shape[:2]
    img_area = height * width
    primitives = []
    region_id_counter = 0

    for idx, color in enumerate(palette):
        mask = np.all(img_data == color, axis=-1).astype(np.uint8)
        labeled_array, _ = nd_label(mask)

        for region in regionprops(labeled_array):
            if region.area < min_region_area or region.area > 0.95 * img_area:
                continue

            region_mask = (labeled_array == region.label).astype(np.uint8)
            contours = find_contours(region_mask, level=0.5)
            if not contours:
                continue

            minr, minc, maxr, maxc = region.bbox
            outlines = []
            for contour in contours:
                dense = interpolate_contour(contour, step=0.5)
                outline = [(int(xf), int(yf)) for yf, xf in dense if 0 <= int(xf) < width and 0 <= int(yf) < height]
                if outline:
                    outlines.append(outline)

            local_spacing = tile_spacing or max(8, min(maxc - minc, maxr - minr) // 4)
            local_font_size = font_size or int(local_spacing * 0.7)
            local_font_size = max(8, min(local_font_size, 24))
            labels = []

            if label_mode == "centroid":
                cx, cy = int(region.centroid[1]), int(region.centroid[0])
                if 0 <= cx < width and 0 <= cy < height:
                    labels = [make_label(cx, cy, idx, local_font_size)]
            elif label_mode == "stable":
                sx, sy = find_stable_label_pixel(region_mask)
                x, y = sx + minc, sy + minr
                if 0 <= x < width and 0 <= y < height:
                    labels = [make_label(x, y, idx, local_font_size)]
            elif label_mode == "diagonal":
                for y in range(minr, min(maxr, height), local_spacing):
                    for x in range(minc, min(maxc, width), local_spacing):
                        if labeled_array[y, x] == region.label:
                            labels.append(make_label(x, y, idx, local_font_size))
                            if x + local_spacing // 2 < width and y + local_spacing // 2 < height:
                                labels.append(make_label(x, y, idx, local_font_size))

            primitives.append(dict(
                outline=outlines,
                labels=labels,
                region_id=region_id_counter,
                color=tuple(int(c) for c in color),
                palette_index=idx,
                bbox=region.bbox
            ))
            region_id_counter += 1

    return primitives
