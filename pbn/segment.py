from skimage.measure import regionprops, find_contours
from skimage.measure import label as sklabel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import typer
import random
from scipy import ndimage as ndi # Added for blobbify_primitives

# --- make_label (already correctly updated by you) ---
def make_label(x, y, value, font_size, region_area):
    return {
        "position": (x, y),
        "value": str(value),
        "font_size": font_size,
        "region_area": region_area
    }

# --- Helper for Bounding Box Overlap (already included by you) ---
def do_bboxes_overlap(box1, box2, margin=1): # box format: (x1, y1, x2, y2)
    return not (box1[2] + margin < box2[0] or
                box1[0] - margin > box2[2] or
                box1[3] + margin < box2[1] or
                box1[1] - margin > box2[3])

# --- Utility 1: Load Font (already included by you) ---
def get_font_for_label(label_data, font_path_str, default_font_size_from_cli):
    font_sz = label_data.get("font_size", default_font_size_from_cli)
    font_to_use = None
    if font_path_str and os.path.isfile(font_path_str):
        try:
            font_to_use = ImageFont.truetype(font_path_str, font_sz)
        except IOError:
            pass
    if font_to_use is None:
        try:
            font_to_use = ImageFont.load_default(size=font_sz)
        except TypeError:
            font_to_use = ImageFont.load_default()
        except AttributeError:
            font_to_use = ImageFont.load_default()
    return font_to_use

# --- Utility 2: Calculate Label Bounding Box (already included by you) ---
def calculate_label_screen_bbox(label_data, font_object, dummy_draw_context, additional_nudge_pixels_up=0):
    lx, ly = label_data["position"]
    text_val = str(label_data["value"])
    try:
        bbox_at_origin = dummy_draw_context.textbbox((0, 0), text_val, font=font_object)
        text_w = bbox_at_origin[2] - bbox_at_origin[0]
        text_h = bbox_at_origin[3] - bbox_at_origin[1]
    except AttributeError:
        font_sz = label_data.get("font_size", 10)
        text_w = font_sz * len(text_val) * 0.6
        text_h = font_sz
        # typer.secho(f"Warning: Using rough estimate for label size for '{text_val}' due to textbbox issue.", fg=typer.colors.YELLOW, err=True)

    effective_y_center = ly - additional_nudge_pixels_up
    x1 = lx - text_w // 2
    y1 = effective_y_center - text_h // 2
    x2 = x1 + text_w
    y2 = y1 + text_h
    return (x1, y1, x2, y2)

# --- Main Collision Resolution Function (already included by you) ---
def resolve_label_collisions(
    primitives_list,
    font_path_str,
    default_font_size_for_fallback,
    additional_nudge_pixels_up=0,
    strategy="fewest_neighbors_then_area",
    neighbor_radius_factor=3.0
):
    all_labels_augmented = []
    dummy_image = Image.new("RGB", (1, 1), (255, 255, 255)) # Corrected: was "L", (1,1)
    draw_context_for_bbox_calc = ImageDraw.Draw(dummy_image) # Renamed for clarity

    label_id_counter = 0
    for prim_idx, primitive in enumerate(primitives_list):
        palette_idx_for_prim = primitive["palette_index"]
        for label_data in primitive["labels"]:
            label_data["id"] = label_id_counter
            label_data["palette_index_for_collision"] = palette_idx_for_prim # Use a distinct key

            font_obj = get_font_for_label(label_data, font_path_str, default_font_size_for_fallback)
            
            bbox_coords = calculate_label_screen_bbox(
                label_data, 
                font_obj, 
                draw_context_for_bbox_calc, # Use the created dummy draw context
                additional_nudge_pixels_up
            )
            # Store bbox directly as a tuple or split for clarity if preferred
            label_data["bbox_x1"], label_data["bbox_y1"], label_data["bbox_x2"], label_data["bbox_y2"] = bbox_coords
            
            all_labels_augmented.append(label_data)
            label_id_counter += 1

    for i, l1 in enumerate(all_labels_augmented):
        l1["neighbor_count"] = 0
        radius = l1["font_size"] * neighbor_radius_factor
        for j, l2 in enumerate(all_labels_augmented):
            if i == j:
                continue
            # Ensure 'palette_index_for_collision' exists before accessing
            if l1.get("palette_index_for_collision") == l2.get("palette_index_for_collision"):
                dist = np.hypot(
                    l1["position"][0] - l2["position"][0],
                    l1["position"][1] - l2["position"][1]
                )
                if dist < radius:
                    l1["neighbor_count"] += 1
    
    if strategy == "smallest_area_only":
        sort_key = lambda l: l["region_area"]
    elif strategy == "fewest_neighbors_only":
        sort_key = lambda l: l.get("neighbor_count", float('inf')) # Handle if key missing
    elif strategy == "smallest_area_then_neighbors":
        sort_key = lambda l: (l["region_area"], l.get("neighbor_count", float('inf')))
    else: # Default: "fewest_neighbors_then_area"
        sort_key = lambda l: (l.get("neighbor_count", float('inf')), l["region_area"])
        
    sorted_labels = sorted(all_labels_augmented, key=sort_key)

    final_kept_label_ids = set()
    placed_bboxes = []

    for label_candidate in sorted_labels:
        candidate_bbox = (
            label_candidate["bbox_x1"], label_candidate["bbox_y1"],
            label_candidate["bbox_x2"], label_candidate["bbox_y2"]
        )
        has_collision = False
        for placed_bbox in placed_bboxes:
            if do_bboxes_overlap(candidate_bbox, placed_bbox, margin=1):
                has_collision = True
                break
        
        if not has_collision:
            final_kept_label_ids.add(label_candidate["id"])
            placed_bboxes.append(candidate_bbox)
            
    for primitive in primitives_list:
        kept_labels_for_primitive = []
        for l in primitive["labels"]:
            if l.get("id") in final_kept_label_ids:
                # Clean up temporary keys
                l.pop("palette_index_for_collision", None)
                l.pop("neighbor_count", None)
                l.pop("bbox_x1", None); l.pop("bbox_y1", None)
                l.pop("bbox_x2", None); l.pop("bbox_y2", None)
                # l.pop("id", None) # Keep 'id' if it's meant to be persistent, otherwise remove
                kept_labels_for_primitive.append(l)
        primitive["labels"] = kept_labels_for_primitive
        
    return primitives_list

# --- Other existing functions (find_stable_label_pixel, interpolate_contour, render_raster_from_primitives, blobbify_region) ---
# These are assumed to be mostly correct from your provided file, with minor corrections below if needed.

def find_stable_label_pixel(region_mask): # (Assumed correct from your file)
    h, w = region_mask.shape
    best_score, best_coord = -1, (0, 0)
    def same_count(x, y, dx, dy):
        count = -1
        while 0 <= x < w and 0 <= y < h and region_mask[y, x]:
            count += 1; x += dx; y += dy
        return count
    ys, xs = np.nonzero(region_mask)
    if not ys.size: return (w//2, h//2) # Fallback for empty mask
    for x, y in zip(xs, ys):
        score = (same_count(x, y, -1, 0) * same_count(x, y, 1, 0) *
                 same_count(x, y, 0, -1) * same_count(x, y, 0, 1))
        if score > best_score:
            best_score = score; best_coord = (x, y)
    return best_coord

def interpolate_contour(contour, step=0.5): # (Assumed correct from your file)
    dense = [];
    if not contour: return dense
    for i in range(len(contour) -1): # Fixed: -1 for pairs
        x0, y0 = contour[i]; x1, y1 = contour[i+1]
        dist = np.hypot(x1-x0, y1-y0)
        num_steps = max(1, int(dist/step))
        for j in range(num_steps): # Iterate up to num_steps - 1
            t = j / float(num_steps) # Ensure float division
            x = x0 + t * (x1-x0); y = y0 + t * (y1-y0)
            dense.append((x,y))
    if contour: dense.append(contour[-1]) # Ensure last point is added
    return dense


def render_raster_from_primitives(canvas_size, primitives, font_path=None, additional_nudge_pixels_up=0): # Added nudge
    width, height = canvas_size
    output = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output)
    pbn_color = (102, 204, 255) # Consider making this configurable

    font_path_str = str(font_path) if font_path else None

    for region in primitives:
        # Draw outlines (assuming this part is as intended)
        for contour_points in region.get("outline", []): # Renamed 'contour' to 'contour_points'
            # For smoother outlines, consider draw.line(contour_points, fill=pbn_color, width=1)
            # If contour_points has enough points (e.g., >1)
            if len(contour_points) > 1:
                 # Pillow's draw.line can take a list of tuples
                 draw.line(contour_points, fill=pbn_color, width=1)
            elif contour_points: # Single point contour
                 draw.point(contour_points[0], fill=pbn_color)


        for label in region["labels"]:
            font_to_use = get_font_for_label(label, font_path_str, label["font_size"]) # Use utility
            
            lx, ly = label["position"]
            text_value = str(label["value"])
            
            effective_y_for_anchor = ly - additional_nudge_pixels_up

            try:
                draw.text((lx, effective_y_for_anchor), text_value, 
                          font=font_to_use, fill=pbn_color, anchor="mm")
            except (TypeError, AttributeError, ValueError): 
                bbox_calc_dummy_img = Image.new("L",(1,1)) # Need a dummy for bbox calc here
                bbox_calc_draw = ImageDraw.Draw(bbox_calc_dummy_img)
                l_bbox_x1, l_bbox_y1, l_bbox_x2, l_bbox_y2 = calculate_label_screen_bbox(
                    label, font_to_use, bbox_calc_draw, additional_nudge_pixels_up
                )
                # Fallback uses absolute top-left
                draw.text((l_bbox_x1, l_bbox_y1), text_value, 
                          font=font_to_use, fill=pbn_color)
                          
    return output

def blobbify_primitives(primitives, img_shape, min_blob_area, max_blob_area, fixed_font_size, interpolate_contours=True):
    h, w = img_shape
    new_primitives = []
    processed_masks_for_blobs = [] # Renamed from 'masks' to avoid confusion
    region_id_counter = 0

    for region_primitive in primitives: # Renamed 'region' to avoid conflict with skimage.regionprops 'region'
        color = region_primitive["color"]
        outlines = region_primitive.get("outline", [])
        if not outlines: continue

        mask_img = Image.new("1", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask_img)
        for contour in outlines:
            if contour and len(contour) > 1: # Ensure contour has points for polygon
                draw_mask.polygon(contour, outline=1, fill=1)
        
        current_region_mask = np.array(mask_img, dtype=np.uint8)
        if not np.any(current_region_mask): continue # Skip if mask is empty

        blobs = blobbify_region(current_region_mask, min_blob_area, max_blob_area)
        for blob_mask_array in blobs: # Renamed 'blob_mask'
            if not np.any(blob_mask_array): continue

            sub_labeled_array = sklabel(blob_mask_array, connectivity=1) # Use connectivity=1 for 4-connectivity
            for sub_region_props in regionprops(sub_labeled_array): # Renamed 'subregion'
                area = sub_region_props.area
                # Create a mask for the current sub_region_props relative to sub_labeled_array
                current_sub_mask = (sub_labeled_array == sub_region_props.label).astype(np.uint8)
                if area < 1: continue

                mask_entry = {
                    "mask": current_sub_mask, "area": area, "color": color,
                    "palette_index": region_primitive["palette_index"], # Use from original primitive
                    "region_id": region_id_counter, # This ID is for the blob
                }
                processed_masks_for_blobs.append(mask_entry)
                region_id_counter += 1
    
    # PASS 2: Merge small blobs (logic seems complex, assuming it's mostly as intended for now)
    # This part might need careful review if issues arise with blobbify
    merged_blobs = []
    final_kept_blobs = [] # Renamed from 'kept'
    used_blob_ids = set() # Renamed from 'used_ids'

    # First pass: keep blobs that are already large enough
    temp_small_blobs = []
    for blob_info in processed_masks_for_blobs: # Renamed 'm'
        if blob_info["area"] >= min_blob_area:
            final_kept_blobs.append(blob_info)
            used_blob_ids.add(blob_info["region_id"]) # Mark as "kept as is"
        else:
            temp_small_blobs.append(blob_info)
            
    # Second pass: try to merge small blobs
    for small_blob_info in temp_small_blobs: # Renamed 'm'
        if small_blob_info["region_id"] in used_blob_ids: # Already merged or kept
            continue

        current_mask = small_blob_info["mask"]
        dilated_mask = ndi.binary_dilation(current_mask, iterations=1) # Ensure ndi is imported
        
        best_match_target = None
        # Prefer merging with already kept large blobs of the same color
        # Then other large blobs, then other small blobs of same color.
        
        # Check against final_kept_blobs (larger ones) first
        potential_merge_targets = []
        for target_blob_info in final_kept_blobs + [b for b in temp_small_blobs if b["region_id"] != small_blob_info["region_id"] and b["region_id"] not in used_blob_ids]:
            if target_blob_info["region_id"] == small_blob_info["region_id"]: continue # Don't merge with self
            if np.any(dilated_mask & target_blob_info["mask"]): # If they overlap/touch
                potential_merge_targets.append(target_blob_info)

        same_color_targets = [t for t in potential_merge_targets if t["palette_index"] == small_blob_info["palette_index"]]
        other_color_targets = [t for t in potential_merge_targets if t["palette_index"] != small_blob_info["palette_index"]]

        if same_color_targets: # Prioritize same color
            best_match_target = max(same_color_targets, key=lambda t: t["area"]) # Merge with largest same-color neighbor
        elif other_color_targets:
            best_match_target = max(other_color_targets, key=lambda t: t["area"]) # Merge with largest other-color neighbor

        if best_match_target:
            # Perform the merge: add small_blob_info's mask to best_match_target's mask
            # Ensure best_match_target is one of the entries in final_kept_blobs or temp_small_blobs
            # This logic needs to correctly update the target blob that might still be in temp_small_blobs
            # or already in final_kept_blobs.
            
            # Find the actual object to update
            target_to_update = None
            if best_match_target["region_id"] in [k["region_id"] for k in final_kept_blobs]:
                 target_to_update = next(k for k in final_kept_blobs if k["region_id"] == best_match_target["region_id"])
            elif best_match_target["region_id"] in [k["region_id"] for k in temp_small_blobs]:
                 target_to_update = next(k for k in temp_small_blobs if k["region_id"] == best_match_target["region_id"])
            
            if target_to_update:
                target_to_update["mask"] = ((target_to_update["mask"] | current_mask) > 0).astype(np.uint8)
                target_to_update["area"] = target_to_update["mask"].sum()
                used_blob_ids.add(small_blob_info["region_id"]) # Mark small blob as merged
            else: # Should not happen if best_match_target was found
                final_kept_blobs.append(small_blob_info) # Failsafe: keep it if target not found for update
                used_blob_ids.add(small_blob_info["region_id"])

        else: # No neighbor to merge with, keep it if it wasn't used.
            if small_blob_info["region_id"] not in used_blob_ids:
                 final_kept_blobs.append(small_blob_info)
                 used_blob_ids.add(small_blob_info["region_id"])
    
    # Filter out any blobs that were merged into others and might still be in final_kept_blobs by mistake
    final_blobs_for_primitives = [b for b in final_kept_blobs if b["mask"].sum() >= min_blob_area]


    for blob_data in final_blobs_for_primitives: # Renamed 'm' to 'blob_data'
        current_blob_mask = blob_data["mask"]
        if not np.any(current_blob_mask): continue

        contours = find_contours(current_blob_mask, level=0.5)
        if not contours: continue

        dense_outlines = []
        for contour_path in contours: # Renamed 'contour' to 'contour_path'
            flipped_path = [(x, y) for y, x in contour_path]
            dense_path = interpolate_contour(flipped_path, step=0.5) if interpolate_contours else flipped_path
            outline_coords = [(int(xf), int(yf)) for xf, yf in dense_path if 0 <= int(xf) < w and 0 <= int(yf) < h]
            if outline_coords and len(outline_coords) > 1: # Ensure valid outline
                dense_outlines.append(outline_coords)
        
        if not dense_outlines: continue

        # Label placement for the blob
        sx, sy = find_stable_label_pixel(current_blob_mask) # sx, sy are local to the mask if it's full image size
        
        # make_label needs region_area, which for a blob is blob_data["area"]
        label_obj = make_label(
            sx, sy, # These are absolute if current_blob_mask is full image size
            blob_data["palette_index"], 
            fixed_font_size, # Use the passed fixed_font_size for blobs
            region_area=blob_data["area"] # Corrected: use blob's area
        )
        new_primitives.append({
            "outline": dense_outlines,
            "labels": [label_obj],
            "region_id": f'blob_{blob_data["region_id"]}',
            "color": blob_data["color"],
            "palette_index": blob_data["palette_index"],
            # "bbox" could be added here if needed, from blob_mask. Bbox of blob_data is not directly from regionprops.
        })
    return new_primitives

# --- Main Primitive Collection Function ---
def collect_region_primitives(
        input_path, palette, font_size=None, font_path=None, tile_spacing=None, 
        min_region_area=50,
        label_mode="diagonal",
        small_region_label_mode="stable", # For small region fallback strategy
        additional_nudge_pixels_up=0, # For consistent label rendering and bbox calculation
        interpolate_contours=True
        ):
    image = Image.open(input_path).convert("RGB")
    img_data = np.array(image)
    height, width = img_data.shape[:2]
    primitives = []
    region_id_counter = 0

    # Create dummy draw object ONCE for all bbox calculations within this function
    dummy_img_for_bbox_calc = Image.new("L", (1,1)) 
    dummy_draw_context = ImageDraw.Draw(dummy_img_for_bbox_calc)
    
    font_path_str = str(font_path) if font_path else None # Convert Path to str for os.path.isfile

    # --- Dynamic Minimum Area Calculation (from your existing code) ---
    base_font_size_for_calc = font_size if font_size is not None else 12
    base_font_size_for_calc = max(8, min(base_font_size_for_calc, 36))
    
    font_for_measurement = None
    if font_path_str and os.path.isfile(font_path_str):
        try: font_for_measurement = ImageFont.truetype(font_path_str, base_font_size_for_calc)
        except IOError: pass # typer.secho warning handled in pbnpy
    if font_for_measurement is None:
        try: font_for_measurement = ImageFont.load_default(size=base_font_size_for_calc)
        except TypeError: font_for_measurement = ImageFont.load_default()
    
    representative_text = "8"
    if palette is not None and len(palette) > 0: # Simplified representative text logic
        max_idx = len(palette) -1
        if max_idx < 10: representative_text = "8"
        elif max_idx < 100: representative_text = "88"
        else: representative_text = "888"

    try:
        text_bbox_m = dummy_draw_context.textbbox((0,0), representative_text, font=font_for_measurement)
        label_w_est = text_bbox_m[2] - text_bbox_m[0]
        label_h_est = text_bbox_m[3] - text_bbox_m[1]
    except AttributeError:
        label_w_est = base_font_size_for_calc * 0.6 * len(representative_text)
        label_h_est = base_font_size_for_calc
        # typer.secho warning handled in pbnpy if necessary

    label_w_est = max(1, label_w_est); label_h_est = max(1, label_h_est)
    dynamic_min_area_for_font = (label_w_est * label_h_est) * 2.5
    dynamic_min_area_for_font = max(dynamic_min_area_for_font, 25)
    actual_min_filter_area = max(min_region_area, int(dynamic_min_area_for_font))
    # typer.echo(...) for this info can be in pbnpy.py

    # --- Iterate through palette colors and then regions ---
    for idx, color_rgb_val in enumerate(palette): # Renamed 'color' to 'color_rgb_val'
        mask = np.all(img_data == color_rgb_val, axis=-1).astype(np.uint8)
        labeled_array = sklabel(mask, connectivity=1) # Use connectivity=1 for 4-way

        for region_props in regionprops(labeled_array): # Renamed 'region' to 'region_props'
            if region_props.area < actual_min_filter_area: # Use effective filter area
                continue
            # Skip very large regions (e.g. background) - from your existing code
            if region_props.area > 0.95 * (width * height) : 
                continue

            # region_mask is for the current specific component
            region_mask = (labeled_array == region_props.label).astype(np.uint8)
            
            contours = find_contours(region_mask, level=0.5)
            if not contours: continue

            minr, minc, maxr, maxc = region_props.bbox
            outlines = []
            for contour_path in contours: # Renamed 'contour'
                # Ensure contour_path is not empty
                if not contour_path.size: continue
                flipped_path = [(x, y) for y, x in contour_path] # skimage gives (row,col) so (y,x)
                dense_path = interpolate_contour(flipped_path, step=0.5) if interpolate_contours else flipped_path
                outline_coords = [(int(xf), int(yf)) for xf, yf in dense_path if 0 <= int(xf) < width and 0 <= int(yf) < height]
                if outline_coords and len(outline_coords) > 1: # Need at least 2 points for a line/polygon segment
                    outlines.append(outline_coords)

            if not outlines: continue # If no valid outlines were formed

            # Use the main font_size passed from pbnpy (effective_font_size)
            local_font_size = font_size if font_size is not None else 12 

            local_spacing_for_diagonal = tile_spacing or max(8, min(maxc - minc, maxr - minr) // 4)
            local_spacing_for_diagonal = max(1, local_spacing_for_diagonal) # Ensure spacing is at least 1

            labels = [] # Initialize labels for current region_props
            region_width = maxc - minc
            region_height = maxr - minr
            
            use_fallback_strategy = (region_width < local_spacing_for_diagonal or 
                                   region_height < local_spacing_for_diagonal)
            
            current_label_mode = label_mode
            if use_fallback_strategy:
                current_label_mode = small_region_label_mode # Override with small region strategy
                # typer.echo(f"Region {idx} small, using fallback: '{current_label_mode}'")

            if current_label_mode == "stable":
                # For stable, coordinates are local to region_mask, convert to global
                sx_local, sy_local = find_stable_label_pixel(region_mask) # This returns coords within region_mask
                # No need for minc, minr offset if region_mask is full image size with only current region active
                # However, region_mask is created based on labeled_array == region_props.label,
                # so it IS a full-image-sized mask. find_stable_label_pixel returns global coords in this case.
                labels = [make_label(sx_local, sy_local, idx, local_font_size, region_area=region_props.area)]
            elif current_label_mode == "centroid":
                # region_props.centroid is (row, col) -> (y, x)
                cy_global, cx_global = int(region_props.centroid[0]), int(region_props.centroid[1])
                labels = [make_label(cx_global, cy_global, idx, local_font_size, region_area=region_props.area)]
            elif current_label_mode == "diagonal":
                row_is_offset = False
                for y_coord in range(minr, maxr, local_spacing_for_diagonal): # Iterate within bbox
                    current_x_offset = local_spacing_for_diagonal // 2 if row_is_offset else 0
                    for x_base_grid in range(minc, maxc, local_spacing_for_diagonal): # Iterate within bbox
                        actual_x_coord = x_base_grid + current_x_offset
                        if not (0 <= actual_x_coord < width and 0 <= y_coord < height): continue
                        
                        # Check against region_mask, not labeled_array[y,x]==specific_label
                        # because labeled_array might have multiple regions of same color if not pre-filtered
                        # region_mask is specific to this one connected component.
                        if region_mask[y_coord, actual_x_coord]:
                            labels.append(make_label(actual_x_coord, y_coord, idx, local_font_size, region_area=region_props.area))
                    row_is_offset = not row_is_offset
            elif current_label_mode == "none": # For small_region_label_mode == "none"
                labels = []

            # --- New Elision Logic for labels >25% outside (if multiple labels for this region) ---
            if len(labels) > 1:
                surviving_labels_for_this_region = []
                for label_candidate in labels:
                    font_obj_for_elision = get_font_for_label(label_candidate, font_path_str, font_size)
                    
                    l_bbox_x1, l_bbox_y1, l_bbox_x2, l_bbox_y2 = calculate_label_screen_bbox(
                        label_candidate, 
                        font_obj_for_elision, 
                        dummy_draw_context, # Use the one created at the start of the function
                        additional_nudge_pixels_up # Use the parameter passed to this function
                    )

                    pixels_in_bbox_total = 0
                    pixels_in_bbox_and_region_mask = 0
                    
                    scan_x_start = max(0, int(np.floor(l_bbox_x1)))
                    scan_x_end = min(width, int(np.ceil(l_bbox_x2))) # Use global width
                    scan_y_start = max(0, int(np.floor(l_bbox_y1)))
                    scan_y_end = min(height, int(np.ceil(l_bbox_y2))) # Use global height

                    for y_scan in range(scan_y_start, scan_y_end):
                        for x_scan in range(scan_x_start, scan_x_end):
                            if (l_bbox_x1 <= x_scan < l_bbox_x2 and
                                l_bbox_y1 <= y_scan < l_bbox_y2):
                                pixels_in_bbox_total += 1
                                if region_mask[y_scan, x_scan]: # region_mask is specific to current component
                                    pixels_in_bbox_and_region_mask += 1
                    
                    percentage_outside = 100.0
                    if pixels_in_bbox_total > 0:
                        percentage_outside = (1.0 - (pixels_in_bbox_and_region_mask / float(pixels_in_bbox_total))) * 100.0
                    
                    if percentage_outside <= 25.0:
                        surviving_labels_for_this_region.append(label_candidate)
                    else:
                        # Optional: typer.secho for debugging elided labels
                        pass 
                        # typer.secho(f"Label '{label_candidate['value']}' for region {idx} elided: {percentage_outside:.1f}% outside (multi-labels).", fg=typer.colors.YELLOW, err=True)
                
                labels = surviving_labels_for_this_region
            # --- End of New Elision Logic ---
            
            if outlines and labels: # Only add primitive if it has outlines and (after elision) labels
                 primitives.append(dict(
                    outline=outlines,
                    labels=labels, # This is now the potentially filtered list
                    region_id=region_id_counter, # Unique ID for this processed region part
                    color=tuple(int(c) for c in color_rgb_val),
                    palette_index=idx, # Palette index
                    # bbox=region_props.bbox # Bbox of the region_props object
                ))
            region_id_counter += 1 # Increment for each processed region_props that results in a primitive

    return primitives