from skimage.measure import regionprops, find_contours
from skimage.measure import label as sklabel
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np # keep np for type hints
import os
import random
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Optional, Union 
from numba import cuda
import math
from pathlib import Path

_PBNPY_FORCE_NUMPY_ENV = os.environ.get("PBNPY_FORCE_NUMPY", "0").lower()
FORCE_NUMPY_BACKEND = _PBNPY_FORCE_NUMPY_ENV in ["1", "true", "yes"]

if FORCE_NUMPY_BACKEND:
    xp = np 
    import scipy.ndimage 
    ndi_xp = scipy.ndimage
    GPU_ENABLED = False
else:
    try:
        import cupy as xp
        import cupyx.scipy.ndimage as ndi_xp
        if xp.cuda.is_available():
            try:
                GPU_ENABLED 
            except NameError:
                GPU_ENABLED = True
        else:
            import numpy as xp_fallback 
            import scipy.ndimage as ndi_xp_fallback
            xp = xp_fallback
            ndi_xp = ndi_xp_fallback
            GPU_ENABLED = False
    except ImportError:
        import numpy as xp_fallback
        import scipy.ndimage as ndi_xp_fallback
        xp = xp_fallback
        ndi_xp = ndi_xp_fallback
        GPU_ENABLED = False

@cuda.jit
def scan_rows_kernel_for_gpu(matrix_in_gpu, counts_out_gpu, num_cols):
    row_idx = cuda.grid(1)
    if row_idx < matrix_in_gpu.shape[0]:
        if num_cols > 0:
            counts_out_gpu[row_idx, num_cols - 1] = matrix_in_gpu[row_idx, num_cols - 1]
            for i in range(num_cols - 2, -1, -1):
                if matrix_in_gpu[row_idx, i] == 1:
                    counts_out_gpu[row_idx, i] = counts_out_gpu[row_idx, i + 1] + 1
                else:
                    counts_out_gpu[row_idx, i] = 0


def blobbify_region(region_mask: xp.ndarray, min_blob_area: int, max_blob_area: int) -> list[xp.ndarray]:
    if not xp.any(region_mask):
        return []
    output_blob_masks = []
    processing_queue = []
    if xp.__name__ == 'cupy':
        numpy_region_mask_for_sklabel = xp.asnumpy(region_mask)
    else:
        numpy_region_mask_for_sklabel = region_mask
    numpy_labeled_array = sklabel(numpy_region_mask_for_sklabel, connectivity=1, background=0)
    labeled_initial_components = xp.asarray(numpy_labeled_array)
    if xp.__name__ == 'cupy':
        numpy_labeled_for_props = xp.asnumpy(labeled_initial_components)
    else:
        numpy_labeled_for_props = labeled_initial_components
    props_initial_components = regionprops(numpy_labeled_for_props)

    for props in props_initial_components:
        if props.area > 0:
            component_mask = (labeled_initial_components == props.label).astype(xp.uint8)
            processing_queue.append(component_mask)

    while processing_queue:
        current_mask = processing_queue.pop(0)
        current_area = xp.sum(current_mask)
        if current_area == 0: continue
        if current_area <= max_blob_area:
            output_blob_masks.append(current_mask)
        else:
            distance = ndi_xp.distance_transform_edt(current_mask)
            min_dist_for_peaks = max(3, int(xp.sqrt(max_blob_area) / 3.0))
            num_desired_blobs = max(2, int(xp.ceil(current_area / max_blob_area)))
            if xp.__name__ == 'cupy':
                distance_for_peak_local_max = xp.asnumpy(distance)
                labels_for_peak_local_max = xp.asnumpy(current_mask)
            else:
                distance_for_peak_local_max = distance
                labels_for_peak_local_max = current_mask
            local_maxi_coords_np = peak_local_max(
                distance_for_peak_local_max,
                min_distance=min_dist_for_peaks,
                labels=labels_for_peak_local_max,
                num_peaks=num_desired_blobs * 2
            )
            local_maxi_coords = xp.asarray(local_maxi_coords_np) 
            if local_maxi_coords.shape[0] < 2:
                output_blob_masks.append(current_mask)
                continue
            markers_bool_mask = xp.zeros(distance.shape, dtype=bool)
            if local_maxi_coords.size > 0: 
                markers_bool_mask[tuple(local_maxi_coords.T)] = True

            markers_labeled, num_marker_features = ndi_xp.label(markers_bool_mask)
            if num_marker_features < 2 :
                output_blob_masks.append(current_mask)
                continue
            if xp.__name__ == 'cupy':
                image_for_watershed_np = xp.asnumpy(-distance)
                markers_for_watershed_np = xp.asnumpy(markers_labeled)
                mask_for_watershed_np = xp.asnumpy(current_mask)
            else:
                image_for_watershed_np = -distance
                markers_for_watershed_np = markers_labeled
                mask_for_watershed_np = current_mask
            connectivity_val = 2
            watershed_segments_np = watershed(
                image_for_watershed_np,
                markers=markers_for_watershed_np,
                connectivity=connectivity_val,
                mask=mask_for_watershed_np
            )
            watershed_segments = xp.asarray(watershed_segments_np)
            unique_segment_labels = xp.unique(watershed_segments)
            num_actual_segments_created = 0
            for seg_label in unique_segment_labels:
                if seg_label == 0: continue
                segment_mask = (watershed_segments == seg_label).astype(xp.uint8)
                if xp.sum(segment_mask) > 0:
                    processing_queue.append(segment_mask)
                    num_actual_segments_created +=1
            if num_actual_segments_created <= 1 and current_area > max_blob_area:
                output_blob_masks.append(current_mask)
    final_masks = [mask for mask in output_blob_masks if xp.any(mask)]
    return final_masks

def make_label(x, y, value, font_size, region_area):
    return {
        "position": (x, y),
        "value": str(value),
        "font_size": font_size,
        "region_area": region_area
    }

def do_bboxes_overlap(box1: tuple[float, float, float, float],
                      box2: tuple[float, float, float, float],
                      margin: float = 1.0) -> bool:
    return not (box1[2] + margin < box2[0] or
                box1[0] - margin > box2[2] or
                box1[3] + margin < box2[1] or
                box1[1] - margin > box2[3])

def get_font_for_label(label_data: dict, font_path_str: Optional[str], default_font_size_from_cli: int) -> ImageFont.FreeTypeFont:
    font_sz = label_data.get("font_size", default_font_size_from_cli)
    font_to_use = None
    if font_path_str and os.path.isfile(font_path_str):
        try:
            font_to_use = ImageFont.truetype(font_path_str, font_sz)
        except IOError:
            pass
    if font_to_use is None:
        try:
            font_to_use = ImageFont.truetype("arial.ttf", font_sz)
        except IOError:
            try:
                font_to_use = ImageFont.load_default(size=font_sz)
            except TypeError:
                font_to_use = ImageFont.load_default()
            except AttributeError:
                 font_to_use = ImageFont.load_default()
    return font_to_use

def calculate_label_screen_bbox(label_data: dict, font_object: ImageFont.FreeTypeFont,
                                dummy_draw_context: ImageDraw.ImageDraw,
                                additional_nudge_pixels_up: float = 0.0) -> tuple[float, float, float, float]:
    lx, ly = label_data["position"]
    text_val = str(label_data["value"])
    text_w, text_h = 0, 0 
    try:
        bbox_at_origin = font_object.getbbox(text_val)
        text_w = bbox_at_origin[2] - bbox_at_origin[0]
        text_h = bbox_at_origin[3] - bbox_at_origin[1]
    except AttributeError:
        try:
            bbox_pil = dummy_draw_context.textbbox((0, 0), text_val, font=font_object)
            text_w = bbox_pil[2] - bbox_pil[0]
            text_h = bbox_pil[3] - bbox_pil[1]
        except AttributeError: 
            font_sz = label_data.get("font_size", 10)
            text_w = font_sz * len(text_val) * 0.6 
            text_h = font_sz 
    effective_y_center = float(ly) - additional_nudge_pixels_up
    x1 = float(lx) - text_w / 2.0
    y1 = effective_y_center - text_h / 2.0
    x2 = x1 + text_w
    y2 = y1 + text_h
    return (x1, y1, x2, y2)

def resolve_label_collisions(
    primitives_list: list[dict],
    font_path_str: Optional[str],
    default_font_size_for_fallback: int,
    additional_nudge_pixels_up: float = 0.0,
    strategy: str = "fewest_neighbors_then_area",
    neighbor_radius_factor: float = 3.0
) -> list[dict]:
    all_labels_initial_data = []
    dummy_image = Image.new("RGB", (1, 1))
    draw_context_for_bbox_calc = ImageDraw.Draw(dummy_image)
    label_id_counter = 0
    for prim_idx, primitive in enumerate(primitives_list):
        palette_idx_for_prim = primitive["palette_index"]
        for label_data in primitive["labels"]:
            label_data["id"] = label_id_counter
            font_obj = get_font_for_label(label_data, font_path_str, default_font_size_for_fallback)
            bbox_coords = calculate_label_screen_bbox(
                label_data, font_obj, draw_context_for_bbox_calc, additional_nudge_pixels_up
            )
            current_label_info = {
                "id": label_id_counter,
                "original_primitive_idx": prim_idx,
                "original_label_obj": label_data,
                "pos_x": float(label_data["position"][0]),
                "pos_y": float(label_data["position"][1]),
                "font_size": float(label_data.get("font_size", default_font_size_for_fallback)),
                "palette_index": palette_idx_for_prim,
                "region_area": float(label_data.get("region_area", 0.0)),
                "bbox_x1": bbox_coords[0], "bbox_y1": bbox_coords[1],
                "bbox_x2": bbox_coords[2], "bbox_y2": bbox_coords[3],
            }
            all_labels_initial_data.append(current_label_info)
            label_id_counter += 1
    if not all_labels_initial_data: return primitives_list
    num_labels = len(all_labels_initial_data)
    xp_ids = xp.array([l["id"] for l in all_labels_initial_data], dtype=int)
    xp_pos_x = xp.array([l["pos_x"] for l in all_labels_initial_data], dtype=float)
    xp_pos_y = xp.array([l["pos_y"] for l in all_labels_initial_data], dtype=float)
    xp_font_sizes = xp.array([l["font_size"] for l in all_labels_initial_data], dtype=float)
    xp_palette_indices = xp.array([l["palette_index"] for l in all_labels_initial_data], dtype=int)
    xp_region_areas = xp.array([l["region_area"] for l in all_labels_initial_data], dtype=float)
    xp_bboxes = xp.array(
        [[l["bbox_x1"], l["bbox_y1"], l["bbox_x2"], l["bbox_y2"]] for l in all_labels_initial_data],
        dtype=float
    )
    dx = xp_pos_x[:, None] - xp_pos_x[None, :]
    dy = xp_pos_y[:, None] - xp_pos_y[None, :]
    distances = xp.hypot(dx, dy)
    radii_for_neighbors = xp_font_sizes * neighbor_radius_factor
    same_palette_mask = (xp_palette_indices[:, None] == xp_palette_indices[None, :])
    identity_matrix = xp.eye(num_labels, dtype=bool) 
    not_self_mask = ~identity_matrix
    is_within_radius_mask = (distances < radii_for_neighbors[:, None])
    is_neighbor_mask = is_within_radius_mask & same_palette_mask & not_self_mask
    xp_neighbor_counts = xp.sum(is_neighbor_mask, axis=1)
    
    keys_for_lexsort: Union[tuple[xp.ndarray, ...], xp.ndarray] 
    if strategy == "smallest_area_only":
        sorted_indices = xp.argsort(xp_region_areas)
    elif strategy == "fewest_neighbors_only":
        sorted_indices = xp.argsort(xp_neighbor_counts)
    elif strategy == "smallest_area_then_neighbors":
        if GPU_ENABLED and hasattr(xp, 'stack'): 
            keys_for_lexsort = xp.stack((xp_neighbor_counts, xp_region_areas), axis=0)
        else: 
            keys_for_lexsort = (xp_neighbor_counts, xp_region_areas)
        sorted_indices = xp.lexsort(keys_for_lexsort)
    else: 
        if GPU_ENABLED and hasattr(xp, 'stack'): 
            keys_for_lexsort = xp.stack((xp_region_areas, xp_neighbor_counts), axis=0)
        else: 
            keys_for_lexsort = (xp_region_areas, xp_neighbor_counts)
        sorted_indices = xp.lexsort(keys_for_lexsort)

    final_kept_label_ids = set()
    placed_bboxes_tuples = []
    sorted_indices_np = xp.asnumpy(sorted_indices) if GPU_ENABLED else sorted_indices
    for i in sorted_indices_np:
        candidate_bbox_coords = tuple(xp_bboxes[i].tolist()) 
        has_collision = False
        for placed_bbox_tuple in placed_bboxes_tuples:
            if do_bboxes_overlap(candidate_bbox_coords, placed_bbox_tuple, margin=1):
                has_collision = True; break
        if not has_collision:
            final_kept_label_ids.add(int(xp_ids[i].item()) if hasattr(xp_ids[i], 'item') else int(xp_ids[i]))
            placed_bboxes_tuples.append(candidate_bbox_coords)
    for primitive_data in primitives_list:
        kept_labels_for_this_primitive = []
        for label_obj_in_primitive in primitive_data["labels"]:
            if label_obj_in_primitive.get("id") in final_kept_label_ids:
                kept_labels_for_this_primitive.append(label_obj_in_primitive)
        primitive_data["labels"] = kept_labels_for_this_primitive
    return primitives_list

def find_stable_label_pixel(region_mask: xp.ndarray) -> tuple[int, int]:
    h, w = region_mask.shape
    if xp.sum(region_mask) == 0: return (w // 2, h // 2)
    def _scan_runs_rowwise_left_to_right(matrix_rows: xp.ndarray) -> xp.ndarray:
        num_rows, num_cols = matrix_rows.shape
        counts = xp.zeros_like(matrix_rows, dtype=xp.int32)
        if num_cols == 0: return counts
        if GPU_ENABLED and xp.__name__ == 'cupy':
            matrix_for_kernel = matrix_rows.astype(xp.uint8) if matrix_rows.dtype == bool else matrix_rows
            threads_per_block = 256
            blocks_per_grid = (num_rows + (threads_per_block - 1)) // threads_per_block
            scan_rows_kernel_for_gpu[blocks_per_grid, threads_per_block](matrix_for_kernel, counts, num_cols)
        else:
            matrix_int = matrix_rows.astype(int)
            counts[:, num_cols - 1] = matrix_int[:, num_cols - 1]
            for i in range(num_cols - 2, -1, -1):
                counts[:, i] = (counts[:, i + 1] + 1) * matrix_int[:, i]
        return counts
    counts_r = _scan_runs_rowwise_left_to_right(region_mask)
    counts_l = xp.fliplr(_scan_runs_rowwise_left_to_right(xp.fliplr(region_mask)))
    mask_T = region_mask.T
    counts_d = _scan_runs_rowwise_left_to_right(mask_T).T
    counts_u = xp.fliplr(_scan_runs_rowwise_left_to_right(xp.fliplr(mask_T))).T
    score_val_r = xp.maximum(0, counts_r - 1)
    score_val_l = xp.maximum(0, counts_l - 1)
    score_val_d = xp.maximum(0, counts_d - 1)
    score_val_u = xp.maximum(0, counts_u - 1)
    scores = score_val_r * score_val_l * score_val_d * score_val_u
    flat_idx = xp.argmax(scores)
    best_y_xp, best_x_xp = xp.unravel_index(flat_idx, scores.shape)
    return (int(best_x_xp.item()), int(best_y_xp.item()))

def interpolate_contour(contour: list[tuple[float,float]], step: float = 0.5) -> list[tuple[float,float]]:
    dense = []
    if not contour or len(contour) < 2 : return contour
    for i in range(len(contour) - 1):
        x0, y0 = contour[i]
        x1, y1 = contour[i+1]
        dist = np.hypot(x1 - x0, y1 - y0)
        num_steps = max(1, int(dist / step))
        for j in range(num_steps):
            t = j / float(num_steps)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            dense.append((x, y))
    dense.append(contour[-1])
    return dense

def render_raster_from_primitives(
    canvas_size: tuple[int, int],
    primitives: list[dict],
    font_path: Optional[Path] = None,
    additional_nudge_pixels_up: float = 0.0,
    label_text_color: str = "#88ddff",
    outline_color_str_hex: str = "#88ddff"
) -> Image.Image:
    width, height = canvas_size
    output_img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output_img)
    
    # Convert hex string outline color to RGB tuple for Pillow
    try:
        outline_render_color_rgb = ImageColor.getrgb(outline_color_str_hex)
    except ValueError:
        # Fallback if the color string is invalid (though Typer might catch some upstream)
        print(f"Warning: Invalid outline color string '{outline_color_str_hex}' for raster. Defaulting to blue.")
        outline_render_color_rgb = (102, 204, 255) # Original hardcoded blue

    font_path_str = str(font_path) if font_path else None

    for region_primitive in primitives:
        # Draw outlines
        for contour_points_list in region_primitive.get("outline", []):
            if len(contour_points_list) > 1:
                 draw.line(contour_points_list, fill=outline_render_color_rgb, width=1) 
            elif contour_points_list:
                 draw.point(contour_points_list[0], fill=outline_render_color_rgb)        

        # Draw text labels
        for label_data in region_primitive["labels"]:
            default_render_font_size = label_data.get("font_size", 10) 
            font_to_use = get_font_for_label(label_data, font_path_str, default_render_font_size)
            lx, ly = label_data["position"]
            text_value = str(label_data["value"])
            
            # This is the y-coordinate around which "mm" anchor should center vertically
            effective_y_center_for_anchor = float(ly) - additional_nudge_pixels_up 
            
            try:
                # Attempt to use "mm" (middle-middle) anchor if available
                draw.text((float(lx), effective_y_center_for_anchor), text_value,
                          font=font_to_use, fill=label_text_color, anchor="mm") 
            except (TypeError, AttributeError, ValueError): # Fallback for older Pillow or if anchor="mm" fails
                # Manually calculate position to simulate middle anchor
                text_width, text_height = 0, 0
                try: # Try getbbox first (Pillow 9.2.0+)
                    bbox_pil = font_to_use.getbbox(text_value)
                    text_width = bbox_pil[2] - bbox_pil[0]
                    text_height = bbox_pil[3] - bbox_pil[1]
                except AttributeError: # Fallback to textbbox (older Pillow)
                    try:
                        bbox_pil = draw.textbbox((0,0), text_value, font=font_to_use) # x,y don't matter for textbbox for size
                        text_width = bbox_pil[2] - bbox_pil[0]
                        text_height = bbox_pil[3] - bbox_pil[1]
                    except AttributeError: # Ultimate fallback if font methods fail
                         font_sz_fallback = label_data.get("font_size", 10)
                         text_width = font_sz_fallback * len(text_value) * 0.6 # Rough estimate
                         text_height = font_sz_fallback # Rough estimate
                
                # Calculate top-left for drawing
                draw_x = float(lx) - (text_width / 2.0)
                draw_y = effective_y_center_for_anchor - (text_height / 2.0)
                
                draw.text((draw_x, draw_y), text_value,
                          font=font_to_use, fill=label_text_color) 
    return output_img

def blobbify_primitives(primitives, img_shape, min_blob_area, max_blob_area, fixed_font_size, interpolate_contours=True):
    h, w = img_shape
    new_primitives = []
    processed_masks_for_blobs = []
    region_id_counter = 0
    for region_primitive in primitives:
        color = region_primitive["color"]
        outlines = region_primitive.get("outline", [])
        if not outlines: continue
        mask_img = Image.new("1", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask_img)
        for contour in outlines:
            if contour and len(contour) > 1:
                draw_mask.polygon(contour, outline=1, fill=1)
        current_region_mask = xp.array(mask_img, dtype=xp.uint8)
        if not xp.any(current_region_mask): continue
        blobs = blobbify_region(current_region_mask, min_blob_area, max_blob_area)
        for blob_mask_array in blobs:
            if not xp.any(blob_mask_array): continue
            if xp.__name__ == 'cupy':
                blob_mask_for_sklabel_np = xp.asnumpy(blob_mask_array)
            else:
                blob_mask_for_sklabel_np = blob_mask_array
            sub_labeled_array_np = sklabel(blob_mask_for_sklabel_np, connectivity=1)
            props_for_loop = regionprops(sub_labeled_array_np)
            for sub_region_props in props_for_loop:
                area = sub_region_props.area
                current_sub_mask_np_comparison = (sub_labeled_array_np == sub_region_props.label)
                current_sub_mask = xp.asarray(current_sub_mask_np_comparison, dtype=xp.uint8)
                if area < 1: continue
                mask_entry = {
                    "mask": current_sub_mask, "area": area, "color": color,
                    "palette_index": region_primitive["palette_index"],
                    "region_id": region_id_counter,
                }
                processed_masks_for_blobs.append(mask_entry)
                region_id_counter += 1
    final_kept_blobs = []
    used_blob_ids = set()
    temp_small_blobs = []
    for blob_info in processed_masks_for_blobs:
        if blob_info["area"] >= min_blob_area:
            final_kept_blobs.append(blob_info)
            used_blob_ids.add(blob_info["region_id"])
        else:
            temp_small_blobs.append(blob_info)
    for small_blob_info in temp_small_blobs:
        if small_blob_info["region_id"] in used_blob_ids: continue
        current_mask_small_blob = small_blob_info["mask"] # Renamed for clarity
        dilated_mask = ndi_xp.binary_dilation(current_mask_small_blob, iterations=1)
        best_match_target = None
        potential_merge_targets = []
        # Iterate over a combined list of already kept large blobs and other small, unused blobs
        candidate_targets = final_kept_blobs + [
            b for b in temp_small_blobs 
            if b["region_id"] != small_blob_info["region_id"] and b["region_id"] not in used_blob_ids
        ]
        for target_blob_info in candidate_targets:
            # No need to check target_blob_info["region_id"] == small_blob_info["region_id"] here due to list comprehension
            if xp.any(dilated_mask & target_blob_info["mask"]):
                potential_merge_targets.append(target_blob_info)
        
        same_color_targets = [t for t in potential_merge_targets if t["palette_index"] == small_blob_info["palette_index"]]
        other_color_targets = [t for t in potential_merge_targets if t["palette_index"] != small_blob_info["palette_index"]]
        if same_color_targets:
            best_match_target = max(same_color_targets, key=lambda t: t["area"])
        elif other_color_targets:
            best_match_target = max(other_color_targets, key=lambda t: t["area"])
        
        if best_match_target:
            target_to_update = None
            # Find the actual list (final_kept_blobs or temp_small_blobs) and update the item
            found_in_kept = False
            for i, k_blob in enumerate(final_kept_blobs):
                if k_blob["region_id"] == best_match_target["region_id"]:
                    final_kept_blobs[i]["mask"] = ((final_kept_blobs[i]["mask"] | current_mask_small_blob) > 0).astype(xp.uint8)
                    final_kept_blobs[i]["area"] = xp.sum(final_kept_blobs[i]["mask"])
                    target_to_update = final_kept_blobs[i] # Reference for logging or further checks
                    found_in_kept = True
                    break
            if not found_in_kept:
                for i, t_blob in enumerate(temp_small_blobs):
                    if t_blob["region_id"] == best_match_target["region_id"]:
                        temp_small_blobs[i]["mask"] = ((temp_small_blobs[i]["mask"] | current_mask_small_blob) > 0).astype(xp.uint8)
                        temp_small_blobs[i]["area"] = xp.sum(temp_small_blobs[i]["mask"])
                        target_to_update = temp_small_blobs[i]
                        break
            
            if target_to_update: # If a merge happened
                used_blob_ids.add(small_blob_info["region_id"]) # Mark small blob as merged
            else: # Failsafe if target somehow wasn't found in lists (should not happen)
                if small_blob_info["region_id"] not in used_blob_ids: 
                    final_kept_blobs.append(small_blob_info)
                    used_blob_ids.add(small_blob_info["region_id"])
        else: # No neighbor to merge with
            if small_blob_info["region_id"] not in used_blob_ids:
                 final_kept_blobs.append(small_blob_info)
                 used_blob_ids.add(small_blob_info["region_id"])

    # Consolidate: start with blobs that were initially large enough or grew large enough in temp_small_blobs
    # Then filter by min_blob_area
    consolidated_blobs = [b for b in final_kept_blobs if b["region_id"] not in used_blob_ids or b["area"] >= min_blob_area]
    for b_info in temp_small_blobs: # Check small blobs that might have grown by merging
        if b_info["region_id"] not in used_blob_ids and b_info["area"] >= min_blob_area:
            consolidated_blobs.append(b_info)
            # No need to add to used_blob_ids here, as this is final collection

    seen_ids_final = set()
    final_blobs_for_primitives = []
    for b in consolidated_blobs: # Renamed from final_blobs_for_primitives_pass1
        current_mask_sum = xp.sum(b["mask"]) # Get current sum
        if b["region_id"] not in seen_ids_final and current_mask_sum >= min_blob_area : 
            b["area"] = current_mask_sum # Ensure area field is up-to-date
            final_blobs_for_primitives.append(b)
            seen_ids_final.add(b["region_id"])

    for blob_data in final_blobs_for_primitives:
        current_blob_mask = blob_data["mask"]
        # Area has been updated just before this loop, so blob_data["area"] is current
        if not xp.any(current_blob_mask) or blob_data["area"] < min_blob_area : continue # Re-check area
        if xp.__name__ == 'cupy':
            current_blob_mask_np = xp.asnumpy(current_blob_mask)
        else:
            current_blob_mask_np = current_blob_mask
        contours_list_np = find_contours(current_blob_mask_np, level=0.5)
        if not contours_list_np: continue
        dense_outlines = []
        for contour_path in contours_list_np:
            if contour_path.ndim != 2 or contour_path.shape[1] != 2: continue 
            flipped_path = [(x, y) for y, x in contour_path] 
            dense_path = interpolate_contour(flipped_path, step=0.5) if interpolate_contours else flipped_path
            outline_coords = [(int(xf), int(yf)) for xf, yf in dense_path if 0 <= int(xf) < w and 0 <= int(yf) < h]
            if outline_coords and len(outline_coords) > 1:
                dense_outlines.append(outline_coords)
        if not dense_outlines: continue
        sx, sy = find_stable_label_pixel(current_blob_mask)
        label_obj = make_label(
            sx, sy, blob_data["palette_index"], fixed_font_size, 
            region_area=blob_data["area"] # Use the updated area
        )
        new_primitives.append({
            "outline": dense_outlines, "labels": [label_obj],
            "region_id": f'blob_{blob_data["region_id"]}', "color": blob_data["color"],
            "palette_index": blob_data["palette_index"],
        })
    return new_primitives

def _check_label_fit_percentage(
    label_to_check: dict,
    region_mask_for_fit: xp.ndarray,
    img_width: int,
    img_height: int,
    font_path_str_for_fit: Optional[str],
    dummy_draw_context_for_fit: ImageDraw.ImageDraw,
    nudge_for_fit: float
) -> float:
    """Helper function to calculate the percentage of a label's bbox outside the region_mask."""
    # Ensure font_size is an int for ImageFont
    font_size_for_check = int(label_to_check.get("font_size", 10)) # Default to 10 if missing
    label_data_for_font = label_to_check.copy()
    label_data_for_font["font_size"] = font_size_for_check # Ensure it's using the int value

    font_obj = get_font_for_label(label_data_for_font, font_path_str_for_fit, font_size_for_check)
    l_bbox_x1, l_bbox_y1, l_bbox_x2, l_bbox_y2 = calculate_label_screen_bbox(
        label_to_check, font_obj, dummy_draw_context_for_fit, nudge_for_fit
    )
    pixels_in_bbox_total = 0
    pixels_in_bbox_and_region_mask = 0
    scan_x_start = max(0, int(math.floor(l_bbox_x1)))
    scan_x_end = min(img_width, int(math.ceil(l_bbox_x2)))
    scan_y_start = max(0, int(math.floor(l_bbox_y1)))
    scan_y_end = min(img_height, int(math.ceil(l_bbox_y2)))

    percentage_outside = 100.0
    if scan_x_start < scan_x_end and scan_y_start < scan_y_end:
        for y_scan in range(scan_y_start, scan_y_end):
            for x_scan in range(scan_x_start, scan_x_end):
                if (l_bbox_x1 <= x_scan + 0.5 < l_bbox_x2 and
                    l_bbox_y1 <= y_scan + 0.5 < l_bbox_y2):
                    pixels_in_bbox_total += 1
                    if region_mask_for_fit[y_scan, x_scan]: # Check on xp array
                        pixels_in_bbox_and_region_mask += 1
        if pixels_in_bbox_total > 0:
            percentage_outside = (1.0 - (pixels_in_bbox_and_region_mask / float(pixels_in_bbox_total))) * 100.0
    return percentage_outside


def collect_region_primitives(
        input_path, palette, font_size=None, font_path: Optional[Path] = None, # Use Path for font_path
        tile_spacing=None,
        min_region_area=50,
        label_strategy="diagonal",
        small_region_label_strategy="stable",
        additional_nudge_pixels_up=0,
        interpolate_contours=True,
        min_font_size_for_scaling: int = 6,
        enable_font_scaling: bool = True # New parameter for toggling
        ):
    image = Image.open(input_path) # No convert("RGB") here, let downstream handle if needed or ensure input is RGB
    img_data = xp.array(image.convert("RGB")) # Convert to RGB before making xp array
    height, width = img_data.shape[:2]
    primitives = []
    region_id_counter = 0
    dummy_img_for_bbox_calc = Image.new("L", (1,1)) # Greyscale is fine for bbox
    dummy_draw_context = ImageDraw.Draw(dummy_img_for_bbox_calc)
    
    # Convert font_path (Path object) to string for os.path.isfile and ImageFont
    font_path_str = str(font_path) if font_path and isinstance(font_path, Path) else None

    base_font_size_for_calc = font_size if font_size is not None else 10
    representative_text = "8"
    if palette is not None and len(palette) > 0:
        max_idx = len(palette) -1 # Max index in the palette
        if max_idx < 10: representative_text = "8"
        elif max_idx < 100: representative_text = "88"
        else: representative_text = "888"

    font_for_measurement = None
    if font_path_str and os.path.isfile(font_path_str): # Check if string path is a file
        try: font_for_measurement = ImageFont.truetype(font_path_str, base_font_size_for_calc)
        except IOError: pass # Font file issue, will fallback
    if font_for_measurement is None: # Fallback to default system font
        try: font_for_measurement = ImageFont.load_default(size=base_font_size_for_calc)
        except TypeError: font_for_measurement = ImageFont.load_default() # Older Pillow
        # except AttributeError: font_for_measurement = ImageFont.load_default() # Pillow 10+

    label_w_est, label_h_est = 0,0 # Initialize
    try:
        if hasattr(font_for_measurement, 'getbbox'): # Pillow 9.2.0+
             # For getbbox, text is drawn starting at (0,0) by default convention for measurement
            text_bbox_m = font_for_measurement.getbbox(representative_text)
        else: # Older Pillow using textbbox
            text_bbox_m = dummy_draw_context.textbbox((0,0), representative_text, font=font_for_measurement)
        label_w_est = text_bbox_m[2] - text_bbox_m[0]
        label_h_est = text_bbox_m[3] - text_bbox_m[1]
    except (AttributeError, TypeError): # Further fallback if font issues
        label_w_est = base_font_size_for_calc * 0.6 * len(representative_text) # Rough estimate
        label_h_est = base_font_size_for_calc # Rough estimate

    label_w_est = max(1, label_w_est); label_h_est = max(1, label_h_est)
    dynamic_min_area_for_font = (label_w_est * label_h_est) * 2.5 # Heuristic factor
    dynamic_min_area_for_font = max(dynamic_min_area_for_font, 25) # Absolute floor for dynamic calc
    
    # actual_min_filter_area is the larger of user-provided min_region_area (or its default 50)
    # and the dynamically calculated area needed for a label.
    actual_min_filter_area = max(min_region_area, int(dynamic_min_area_for_font))

    for idx, color_rgb_val in enumerate(palette):
        color_val_on_device = xp.asarray(color_rgb_val, dtype=img_data.dtype)
        mask = xp.all(img_data == color_val_on_device.reshape(1, 1, 3), axis=-1).astype(xp.uint8)
        if not xp.any(mask): continue

        labeled_array_on_device: xp.ndarray
        num_features: int
        structure_4conn = xp.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
        
        if GPU_ENABLED:
             labeled_array_on_device, num_features = ndi_xp.label(mask, structure=structure_4conn)
        else:
             labeled_array_on_device, num_features = ndi_xp.label(mask, structure=structure_4conn) # SciPy also accepts structure

        if num_features == 0: continue
        
        labeled_array_for_regionprops_np: np.ndarray = xp.asnumpy(labeled_array_on_device) if GPU_ENABLED else labeled_array_on_device
        
        for region_props in regionprops(labeled_array_for_regionprops_np):
            if region_props.area < actual_min_filter_area: continue
            # Skip if region is essentially the whole image (background-like)
            if region_props.area > 0.95 * (width * height) : continue # Example threshold
            
            region_mask = (labeled_array_on_device == region_props.label).astype(xp.uint8)
            
            region_mask_for_contours_np = xp.asnumpy(region_mask) if GPU_ENABLED else region_mask
            contours = find_contours(region_mask_for_contours_np, level=0.5)
            if not contours: continue

            minr, minc, maxr, maxc = region_props.bbox
            outlines: List[List[Tuple[int, int]]] = [] # Type hint
            for contour_path_np in contours: # contour_path is a NumPy array
                if contour_path_np.ndim != 2 or contour_path_np.shape[1] != 2: continue
                # find_contours returns (row, col) which is (y, x)
                flipped_path = [(x, y) for y, x in contour_path_np] 
                dense_path = interpolate_contour(flipped_path, step=0.5) if interpolate_contours else flipped_path
                # Ensure coordinates are integers and within canvas bounds
                outline_coords = [(int(xf), int(yf)) for xf, yf in dense_path if 0 <= int(xf) < width and 0 <= int(yf) < height]
                if outline_coords and len(outline_coords) > 1: # Need at least 2 points for a line
                    outlines.append(outline_coords)
            if not outlines: continue

            local_font_size = font_size if font_size is not None else 10
            local_spacing_for_diagonal = tile_spacing or max(8, min(maxc - minc, maxr - minr) // 4)
            local_spacing_for_diagonal = max(1, local_spacing_for_diagonal) # Ensure at least 1
            
            final_labels_for_primitive: List[dict] = [] # Labels that pass all checks for this primitive
            
            # Determine which labeling strategy to use for this specific region
            region_width_bbox = maxc - minc 
            region_height_bbox = maxr - minr
            use_fallback_strategy = (region_width_bbox < local_spacing_for_diagonal or 
                                   region_height_bbox < local_spacing_for_diagonal)
            current_label_strategy_for_region = label_strategy
            if use_fallback_strategy: 
                current_label_strategy_for_region = small_region_label_strategy

            # --- Generate and Process Labels based on Strategy ---
            if current_label_strategy_for_region == "stable":
                sx_local, sy_local = find_stable_label_pixel(region_mask)
                label_candidate = make_label(sx_local, sy_local, idx, local_font_size, region_props.area)
                
                initial_percentage_outside = _check_label_fit_percentage(
                    label_candidate, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up
                )

                if initial_percentage_outside <= 25.0: # Fits with original font size
                    final_labels_for_primitive.append(label_candidate)
                elif enable_font_scaling: # Doesn't fit, AND font scaling is enabled for stable strategy
                    best_fit_scaled_label = None
                    # Start one step down from local_font_size, down to min_font_size_for_scaling
                    for test_font_size in range(local_font_size - 1, min_font_size_for_scaling - 1, -1):
                        if test_font_size <= 0: continue # Safety, should be caught by min_font_size_for_scaling >=1
                        
                        current_label_attempt = label_candidate.copy() # Use original candidate position
                        current_label_attempt["font_size"] = test_font_size
                        
                        percentage_outside_scaled = _check_label_fit_percentage(
                            current_label_attempt, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up
                        )
                        if percentage_outside_scaled <= 25.0: # Fit condition
                            best_fit_scaled_label = current_label_attempt
                            break # Found a fit, no need to try smaller fonts
                    if best_fit_scaled_label:
                        final_labels_for_primitive.append(best_fit_scaled_label)
                # If enable_font_scaling is False and initial fit failed, label is not added.

            elif current_label_strategy_for_region == "centroid":
                cy_global, cx_global = int(region_props.centroid[0]), int(region_props.centroid[1])
                label_candidate = make_label(cx_global, cy_global, idx, local_font_size, region_props.area)
                percentage_outside = _check_label_fit_percentage(
                    label_candidate, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up
                )
                if percentage_outside <= 25.0:
                    final_labels_for_primitive.append(label_candidate)

            elif current_label_strategy_for_region == "diagonal":
                temp_diagonal_candidates: List[dict] = []
                row_is_offset = False
                for y_coord in range(minr, maxr, local_spacing_for_diagonal):
                    current_x_offset = local_spacing_for_diagonal // 2 if row_is_offset else 0
                    for x_base_grid in range(minc, maxc, local_spacing_for_diagonal):
                        actual_x_coord = x_base_grid + current_x_offset
                        if not (0 <= actual_x_coord < width and 0 <= y_coord < height): continue
                        if region_mask[y_coord, actual_x_coord]: # Check on xp array
                            temp_diagonal_candidates.append(
                                make_label(actual_x_coord, y_coord, idx, local_font_size, region_props.area)
                            )
                    row_is_offset = not row_is_offset
                
                # Apply standard elision to each diagonal candidate (no iterative scaling here)
                for label_candidate_diag in temp_diagonal_candidates:
                    percentage_outside = _check_label_fit_percentage(
                        label_candidate_diag, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up
                    )
                    if percentage_outside <= 25.0:
                        final_labels_for_primitive.append(label_candidate_diag)
            
            # If current_label_strategy_for_region was "none", final_labels_for_primitive remains empty.
            
            if outlines and final_labels_for_primitive: # Check if any labels survived for this region
                 primitives.append(dict(
                    outline=outlines, labels=final_labels_for_primitive, region_id=region_id_counter,
                    color=tuple(int(c) for c in color_rgb_val), palette_index=idx,
                ))
            region_id_counter += 1
    return primitives