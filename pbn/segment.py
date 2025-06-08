from skimage.measure import regionprops, find_contours
from skimage.measure import label as sklabel
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np # keep np for type hints
import os
import random
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Optional, Union, List, Tuple, Dict
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

    try:
        outline_render_color_rgb = ImageColor.getrgb(outline_color_str_hex)
    except ValueError:
        print(f"Warning: Invalid outline color string '{outline_color_str_hex}' for raster. Defaulting to blue.")
        outline_render_color_rgb = (102, 204, 255)

    font_path_str = str(font_path) if font_path else None

    for region_primitive in primitives:
        for contour_points_list in region_primitive.get("outline", []):
            if len(contour_points_list) > 1:
                 draw.line(contour_points_list, fill=outline_render_color_rgb, width=1)
            elif contour_points_list:
                 draw.point(contour_points_list[0], fill=outline_render_color_rgb)

        for label_data in region_primitive["labels"]:
            default_render_font_size = label_data.get("font_size", 10)
            font_to_use = get_font_for_label(label_data, font_path_str, default_render_font_size)
            lx, ly = label_data["position"]
            text_value = str(label_data["value"])

            effective_y_center_for_anchor = float(ly) - additional_nudge_pixels_up

            try:
                draw.text((float(lx), effective_y_center_for_anchor), text_value,
                          font=font_to_use, fill=label_text_color, anchor="mm")
            except (TypeError, AttributeError, ValueError):
                text_width, text_height = 0, 0
                try:
                    bbox_pil = font_to_use.getbbox(text_value)
                    text_width = bbox_pil[2] - bbox_pil[0]
                    text_height = bbox_pil[3] - bbox_pil[1]
                except AttributeError:
                    try:
                        bbox_pil = draw.textbbox((0,0), text_value, font=font_to_use)
                        text_width = bbox_pil[2] - bbox_pil[0]
                        text_height = bbox_pil[3] - bbox_pil[1]
                    except AttributeError:
                         font_sz_fallback = label_data.get("font_size", 10)
                         text_width = font_sz_fallback * len(text_value) * 0.6
                         text_height = font_sz_fallback

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
        current_mask_small_blob = small_blob_info["mask"]
        dilated_mask = ndi_xp.binary_dilation(current_mask_small_blob, iterations=1, brute_force=True)
        best_match_target = None
        potential_merge_targets = []
        candidate_targets = final_kept_blobs + [
            b for b in temp_small_blobs
            if b["region_id"] != small_blob_info["region_id"] and b["region_id"] not in used_blob_ids
        ]
        for target_blob_info in candidate_targets:
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
            found_in_kept = False
            for i, k_blob in enumerate(final_kept_blobs):
                if k_blob["region_id"] == best_match_target["region_id"]:
                    final_kept_blobs[i]["mask"] = ((final_kept_blobs[i]["mask"] | current_mask_small_blob) > 0).astype(xp.uint8)
                    final_kept_blobs[i]["area"] = xp.sum(final_kept_blobs[i]["mask"])
                    target_to_update = final_kept_blobs[i]
                    found_in_kept = True
                    break
            if not found_in_kept:
                for i, t_blob in enumerate(temp_small_blobs):
                    if t_blob["region_id"] == best_match_target["region_id"]:
                        temp_small_blobs[i]["mask"] = ((temp_small_blobs[i]["mask"] | current_mask_small_blob) > 0).astype(xp.uint8)
                        temp_small_blobs[i]["area"] = xp.sum(temp_small_blobs[i]["mask"])
                        target_to_update = temp_small_blobs[i]
                        break

            if target_to_update:
                used_blob_ids.add(small_blob_info["region_id"])
            else:
                if small_blob_info["region_id"] not in used_blob_ids:
                    final_kept_blobs.append(small_blob_info)
                    used_blob_ids.add(small_blob_info["region_id"])
        else:
            if small_blob_info["region_id"] not in used_blob_ids:
                 final_kept_blobs.append(small_blob_info)
                 used_blob_ids.add(small_blob_info["region_id"])

    consolidated_blobs = [b for b in final_kept_blobs if b["region_id"] not in used_blob_ids or b["area"] >= min_blob_area]
    for b_info in temp_small_blobs:
        if b_info["region_id"] not in used_blob_ids and b_info["area"] >= min_blob_area:
            consolidated_blobs.append(b_info)

    seen_ids_final = set()
    final_blobs_for_primitives = []
    for b in consolidated_blobs:
        current_mask_sum = xp.sum(b["mask"])
        if b["region_id"] not in seen_ids_final and current_mask_sum >= min_blob_area :
            b["area"] = current_mask_sum
            final_blobs_for_primitives.append(b)
            seen_ids_final.add(b["region_id"])

    for blob_data in final_blobs_for_primitives:
        current_blob_mask = blob_data["mask"]
        if not xp.any(current_blob_mask) or blob_data["area"] < min_blob_area : continue
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
            region_area=blob_data["area"]
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
    font_size_for_check = int(label_to_check.get("font_size", 10))
    label_data_for_font = label_to_check.copy()
    label_data_for_font["font_size"] = font_size_for_check

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
                    if region_mask_for_fit[y_scan, x_scan]:
                        pixels_in_bbox_and_region_mask += 1
        if pixels_in_bbox_total > 0:
            percentage_outside = (1.0 - (pixels_in_bbox_and_region_mask / float(pixels_in_bbox_total))) * 100.0
    return percentage_outside


def collect_region_primitives(
        input_path, palette, font_size=None, font_path: Optional[Path] = None,
        tile_spacing=None,
        min_region_area=50,
        label_strategy="diagonal",
        small_region_label_strategy="stable",
        additional_nudge_pixels_up=0,
        interpolate_contours=True,
        min_font_size_for_scaling: int = 6,
        enable_font_scaling: bool = True
        ):
    image = Image.open(input_path)
    img_data = xp.array(image.convert("RGB"))
    height, width = img_data.shape[:2]
    # NEW LOGIC: Temporary lists to hold categorized regions
    labeled_regions_data: List[Dict] = []
    unlabeled_regions_to_merge: List[Dict] = []
    
    region_id_counter = 0
    dummy_img_for_bbox_calc = Image.new("L", (1,1))
    dummy_draw_context = ImageDraw.Draw(dummy_img_for_bbox_calc)
    
    font_path_str = str(font_path) if font_path and isinstance(font_path, Path) else None

    base_font_size_for_calc = font_size if font_size is not None else 10
    representative_text = "8"
    if palette is not None and len(palette) > 0:
        max_idx = len(palette) -1
        if max_idx < 10: representative_text = "8"
        elif max_idx < 100: representative_text = "88"
        else: representative_text = "888"

    font_for_measurement = None
    if font_path_str and os.path.isfile(font_path_str):
        try: font_for_measurement = ImageFont.truetype(font_path_str, base_font_size_for_calc)
        except IOError: pass
    if font_for_measurement is None:
        try: font_for_measurement = ImageFont.load_default(size=base_font_size_for_calc)
        except TypeError: font_for_measurement = ImageFont.load_default()

    label_w_est, label_h_est = 0,0
    try:
        if hasattr(font_for_measurement, 'getbbox'):
            text_bbox_m = font_for_measurement.getbbox(representative_text)
        else:
            text_bbox_m = dummy_draw_context.textbbox((0,0), representative_text, font=font_for_measurement)
        label_w_est = text_bbox_m[2] - text_bbox_m[0]
        label_h_est = text_bbox_m[3] - text_bbox_m[1]
    except (AttributeError, TypeError):
        label_w_est = base_font_size_for_calc * 0.6 * len(representative_text)
        label_h_est = base_font_size_for_calc

    label_w_est = max(1, label_w_est); label_h_est = max(1, label_h_est)
    dynamic_min_area_for_font = (label_w_est * label_h_est) * 2.5
    dynamic_min_area_for_font = max(dynamic_min_area_for_font, 25)
    
    actual_min_filter_area = max(min_region_area, int(dynamic_min_area_for_font))

    # NEW LOGIC: Step 1 - Find all regions and attempt to label them, sorting them into two lists.
    for idx, color_rgb_val in enumerate(palette):
        color_val_on_device = xp.asarray(color_rgb_val, dtype=img_data.dtype)
        mask = xp.all(img_data == color_val_on_device.reshape(1, 1, 3), axis=-1).astype(xp.uint8)
        if not xp.any(mask): continue

        structure_4conn = xp.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
        labeled_array_on_device, num_features = ndi_xp.label(mask, structure=structure_4conn)

        if num_features == 0: continue
        
        labeled_array_for_regionprops_np: np.ndarray = xp.asnumpy(labeled_array_on_device) if GPU_ENABLED else labeled_array_on_device
        
        for region_props in regionprops(labeled_array_for_regionprops_np):
            if region_props.area < min_region_area: continue # Use the user-defined minimum here initially
            if region_props.area > 0.95 * (width * height) : continue
            
            region_mask = (labeled_array_on_device == region_props.label).astype(xp.uint8)
            
            final_labels_for_primitive: List[dict] = []
            
            # --- Label generation logic (moved from the end of the loop) ---
            if region_props.area >= actual_min_filter_area:
                local_font_size = font_size if font_size is not None else 10
                local_spacing_for_diagonal = tile_spacing or max(8, min(region_props.bbox[3] - region_props.bbox[1], region_props.bbox[2] - region_props.bbox[0]) // 4)
                local_spacing_for_diagonal = max(1, local_spacing_for_diagonal)

                minr, minc, maxr, maxc = region_props.bbox
                use_fallback_strategy = (maxc - minc < local_spacing_for_diagonal or maxr - minr < local_spacing_for_diagonal)
                current_label_strategy_for_region = small_region_label_strategy if use_fallback_strategy else label_strategy

                if current_label_strategy_for_region == "stable":
                    sx_local, sy_local = find_stable_label_pixel(region_mask)
                    label_candidate = make_label(sx_local, sy_local, idx, local_font_size, region_props.area)
                    
                    if _check_label_fit_percentage(label_candidate, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up) <= 25.0:
                        final_labels_for_primitive.append(label_candidate)
                    elif enable_font_scaling:
                        best_fit_scaled_label = None
                        for test_font_size in range(local_font_size - 1, min_font_size_for_scaling - 1, -1):
                            if test_font_size <= 0: continue
                            current_label_attempt = label_candidate.copy()
                            current_label_attempt["font_size"] = test_font_size
                            if _check_label_fit_percentage(current_label_attempt, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up) <= 25.0:
                                best_fit_scaled_label = current_label_attempt
                                break
                        if best_fit_scaled_label:
                            final_labels_for_primitive.append(best_fit_scaled_label)

                elif current_label_strategy_for_region == "centroid":
                    cy_global, cx_global = int(region_props.centroid[0]), int(region_props.centroid[1])
                    label_candidate = make_label(cx_global, cy_global, idx, local_font_size, region_props.area)
                    if _check_label_fit_percentage(label_candidate, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up) <= 25.0:
                        final_labels_for_primitive.append(label_candidate)

                elif current_label_strategy_for_region == "diagonal":
                    # ... (diagonal strategy remains the same)
                    temp_diagonal_candidates: List[dict] = []
                    row_is_offset = False
                    for y_coord in range(minr, maxr, local_spacing_for_diagonal):
                        current_x_offset = local_spacing_for_diagonal // 2 if row_is_offset else 0
                        for x_base_grid in range(minc, maxc, local_spacing_for_diagonal):
                            actual_x_coord = x_base_grid + current_x_offset
                            if not (0 <= actual_x_coord < width and 0 <= y_coord < height): continue
                            if region_mask[y_coord, actual_x_coord]:
                                temp_diagonal_candidates.append(make_label(actual_x_coord, y_coord, idx, local_font_size, region_props.area))
                        row_is_offset = not row_is_offset
                    for label_candidate_diag in temp_diagonal_candidates:
                        if _check_label_fit_percentage(label_candidate_diag, region_mask, width, height, font_path_str, dummy_draw_context, additional_nudge_pixels_up) <= 25.0:
                            final_labels_for_primitive.append(label_candidate_diag)
            
            # NEW LOGIC: Categorize the region based on whether a label was created.
            region_data = {
                "mask": region_mask,
                "area": region_props.area,
                "color": tuple(int(c) for c in color_rgb_val),
                "palette_index": idx,
                "region_id": region_id_counter,
                "labels": final_labels_for_primitive,
                "merged": False # Flag to track if mask was modified
            }
            
            if final_labels_for_primitive:
                labeled_regions_data.append(region_data)
            else:
                unlabeled_regions_to_merge.append(region_data)
            
            region_id_counter += 1
    
    # NEW LOGIC: Step 2 - Merge unlabeled regions into labeled ones
    # Sort smallest to largest to merge small fragments first
    unlabeled_regions_to_merge.sort(key=lambda r: r['area'])

    for small_region in unlabeled_regions_to_merge:
        dilated_mask = ndi_xp.binary_dilation(small_region['mask'], iterations=2, brute_force=True) # Dilate to find neighbors
        
        potential_merge_targets = []
        for target_region in labeled_regions_data:
            # Check for overlap between dilated small region and a potential target
            if xp.any(dilated_mask & target_region['mask']):
                potential_merge_targets.append(target_region)
        
        if not potential_merge_targets:
            continue # This region is isolated, cannot be merged. It will be dropped.

        # --- Find best merge target (prefer same color, then largest area) ---
        best_match_target = None
        same_color_targets = [t for t in potential_merge_targets if t['palette_index'] == small_region['palette_index']]
        
        if same_color_targets:
            best_match_target = max(same_color_targets, key=lambda t: t['area'])
        else:
            best_match_target = max(potential_merge_targets, key=lambda t: t['area'])

        # --- Perform the merge ---
        # The mask in the list is a reference, so we can update it directly
        best_match_target['mask'] |= small_region['mask']
        best_match_target['area'] = xp.sum(best_match_target['mask']) # Update area
        best_match_target['merged'] = True # Mark that this region was changed

    # NEW LOGIC: Step 3 - Finalize primitives, recalculating outlines and labels for merged regions
    final_primitives: List[Dict] = []
    for region_data in labeled_regions_data:
        # If the region was merged into, its outline and label position are now incorrect. Recalculate them.
        if region_data['merged']:
            # Recalculate stable label position on the new, larger mask
            new_sx, new_sy = find_stable_label_pixel(region_data['mask'])
            
            # Update the first (and likely only) label's position.
            # This assumes 'stable' or 'centroid' like strategies where one label per region is the norm.
            if region_data['labels']:
                # Preserve original font size and value, just update position and area
                original_label = region_data['labels'][0]
                region_data['labels'] = [make_label(new_sx, new_sy, region_data['palette_index'], original_label['font_size'], region_data['area'])]
        
        # --- Generate final outlines for ALL labeled regions ---
        region_mask_for_contours_np = xp.asnumpy(region_data['mask']) if GPU_ENABLED else region_data['mask']
        contours = find_contours(region_mask_for_contours_np, level=0.5)
        if not contours: continue

        outlines: List[List[Tuple[int, int]]] = []
        for contour_path_np in contours:
            if contour_path_np.ndim != 2 or contour_path_np.shape[1] != 2: continue
            flipped_path = [(x, y) for y, x in contour_path_np]
            dense_path = interpolate_contour(flipped_path, step=0.5) if interpolate_contours else flipped_path
            outline_coords = [(int(xf), int(yf)) for xf, yf in dense_path if 0 <= int(xf) < width and 0 <= int(yf) < height]
            if outline_coords and len(outline_coords) > 1:
                outlines.append(outline_coords)
        
        if outlines and region_data['labels']:
            final_primitives.append({
                "outline": outlines,
                "labels": region_data['labels'],
                "region_id": region_data['region_id'],
                "color": region_data['color'],
                "palette_index": region_data['palette_index'],
            })
            
    return final_primitives