from skimage.measure import regionprops, find_contours
from skimage.measure import label as sklabel
from PIL import Image, ImageDraw, ImageFont
import numpy as np # keep np for type hints
import os
import typer
import random
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Optional
from numba import cuda
import math 

_PBNPY_FORCE_NUMPY_ENV = os.environ.get("PBNPY_FORCE_NUMPY", "0").lower()
print(f"DEBUG: _PBNPY_FORCE_NUMPY_ENV = {_PBNPY_FORCE_NUMPY_ENV}")
FORCE_NUMPY_BACKEND = 0
print(f"DEBUG: FORCE_NUMPY_BACKEND = {FORCE_NUMPY_BACKEND}")

if FORCE_NUMPY_BACKEND:
    print("PBNPY_FORCE_NUMPY is set. Forcing NumPy backend in segment.py.")
    xp = np # Alias xp to numpy
    import scipy.ndimage # Alias ndi_xp to scipy.ndimage
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
                print("CuPy found, using GPU in segment.py.")
                GPU_ENABLED = True
        else:
            # Fallback if CUDA not available for CuPy
            print("CuPy found but CUDA not available, falling back to NumPy/SciPy for CPU in segment.py.")
            import numpy as xp
            import scipy.ndimage as ndi_xp
            GPU_ENABLED = False
    except ImportError:
        print("CuPy not found, falling back to NumPy/SciPy for CPU in segment.py.")
        import numpy as xp
        import scipy.ndimage as ndi_xp
        GPU_ENABLED = False

@cuda.jit
def scan_rows_kernel_for_gpu(matrix_in_gpu, counts_out_gpu, num_cols):
    """
    Numba CUDA kernel to perform a right-to-left scan for consecutive 1s on each row.
    Each GPU thread is intended to process one row of matrix_in_gpu.

    Args:
        matrix_in_gpu: 2D device array (e.g., CuPy array) of 0s and 1s (uint8 recommended).
        counts_out_gpu: 2D device array (e.g., CuPy array) to store the resulting counts (int32 recommended).
        num_cols: The number of columns in matrix_in_gpu and counts_out_gpu.
    """
    # cuda.grid(1) gives a unique global index for each thread in a 1D grid.
    # We map this directly to the row index.
    row_idx = cuda.grid(1)

    # Ensure the thread is within the bounds of the number of rows
    if row_idx < matrix_in_gpu.shape[0]:
        if num_cols > 0:
            # Initialize the last element of the current row in counts_out_gpu
            counts_out_gpu[row_idx, num_cols - 1] = matrix_in_gpu[row_idx, num_cols - 1]
            
            # Perform the scan from right-to-left for the current row
            # This loop runs *inside each GPU thread* for its assigned row
            for i in range(num_cols - 2, -1, -1): # Python's range is fine here for Numba
                if matrix_in_gpu[row_idx, i] == 1: # Assuming matrix_in_gpu contains 0s and 1s
                    counts_out_gpu[row_idx, i] = counts_out_gpu[row_idx, i + 1] + 1
                else:
                    counts_out_gpu[row_idx, i] = 0


def blobbify_region(region_mask: xp.ndarray, min_blob_area: int, max_blob_area: int) -> list[xp.ndarray]:
    """
    Breaks down a large binary region mask into smaller "blob" masks.
    Attempts to split components larger than max_blob_area using watershed segmentation.
    The min_blob_area is a guideline; subsequent merging logic in the main
    blobbify_primitives function handles blobs that are too small.

    Args:
        region_mask (np.ndarray): Binary mask of the region to blobbify.
                                  Expected to be 2D.
        min_blob_area (int): Minimum desired area for a blob. This is more of a
                             hint for splitting logic if needed, as final merging
                             handles small blobs.
        max_blob_area (int): Maximum desired area for a blob. Components larger
                             than this will be candidates for splitting.

    Returns:
        list[np.ndarray]: A list of 2D numpy arrays, where each array is a
                          binary mask for a resulting blob. These masks are
                          full-sized (same shape as input region_mask).
    """
    if not xp.any(region_mask):
        return []

    output_blob_masks = []
    
    # Queue will hold full-sized masks that need processing.
    # Each item in the queue is a binary mask of a component.
    processing_queue = []

    # Initial connected components in the input region_mask.
    # These are the starting candidates for blobs or for splitting.

    if xp.__name__ == 'cupy':
        numpy_region_mask_for_sklabel = xp.asnumpy(region_mask)
    else:
        numpy_region_mask_for_sklabel = region_mask # It's already a NumPy array

    # sklabel takes NumPy array, returns NumPy array
    numpy_labeled_array = sklabel(numpy_region_mask_for_sklabel, connectivity=1, background=0)

    # Convert the result to an xp array (CuPy if GPU_ENABLED)
    labeled_initial_components = xp.asarray(numpy_labeled_array)

    if xp.__name__ == 'cupy':
        numpy_labeled_for_props = xp.asnumpy(labeled_initial_components)
    else:
        numpy_labeled_for_props = labeled_initial_components # It's already NumPy
    props_initial_components = regionprops(numpy_labeled_for_props)

    for props in props_initial_components:
        if props.area > 0:
            # Create a full-size mask for this component
            component_mask = (labeled_initial_components == props.label).astype(xp.uint8)
            #if xp.__name__ == 'cupy': 
            #   component_mask = xp.asnumpy(component_mask.get())

            processing_queue.append(component_mask)

    while processing_queue:
        current_mask = processing_queue.pop(0)
        current_area = xp.sum(current_mask)

        if current_area == 0:
            continue

        if current_area <= max_blob_area:
            # If it's small enough (or became small enough after a split), add it.
            # Blobs smaller than min_blob_area are handled by merging logic
            # in the calling function (blobbify_primitives).
            output_blob_masks.append(current_mask)
        else:
            if xp.__name__ == 'cupy':
                print(f"DEBUG: current_mask.device = {current_mask.device}")
            # Component is too large, attempt to split using watershed
            distance = ndi_xp.distance_transform_edt(current_mask)
            
            # Heuristic for min_distance for peak_local_max.
            # Aim for blobs that are roughly circular/square with an area around max_blob_area.
            # The characteristic length would be sqrt(max_blob_area).
            # min_distance between peaks could be a fraction of this length.
            min_dist_for_peaks = max(3, int(xp.sqrt(max_blob_area) / 3.0))

            # Calculate how many blobs we might roughly expect
            num_desired_blobs = max(2, int(xp.ceil(current_area / max_blob_area)))

            if xp.__name__ == 'cupy': # Or your GPU_ENABLED flag
                distance_for_peak_local_max = xp.asnumpy(distance)
                labels_for_peak_local_max = xp.asnumpy(current_mask)
            else: # xp is numpy
                distance_for_peak_local_max = distance
                labels_for_peak_local_max = current_mask

            local_maxi_coords = peak_local_max(
                distance_for_peak_local_max,
                min_distance=min_dist_for_peaks,
                labels=labels_for_peak_local_max, # Process peaks only within the current_mask
                                     # current_mask acts as label image 1 here.
                num_peaks=num_desired_blobs * 2 # Find more peaks than strictly needed as a buffer
            )

            local_maxi_coords = xp.asarray(local_maxi_coords)

            if local_maxi_coords.shape[0] < 2:
                # Not enough distinct peaks for watershed to perform a meaningful split.
                # Add the current large mask as is; it couldn't be split by this method.
                output_blob_masks.append(current_mask)
                continue

            markers_bool_mask = xp.zeros(distance.shape, dtype=bool)
            markers_bool_mask[tuple(local_maxi_coords.T)] = True
            markers_labeled, num_marker_features = ndi_xp.label(markers_bool_mask)

            if num_marker_features < 2 :
                # After labeling markers, still effectively one marker region.
                output_blob_masks.append(current_mask)
                continue

            # Ensure -distance, markers_labeled, and current_mask are NumPy arrays for skimage.watershed
            # ... inside blobbify_region, before watershed call
            # Assume -distance, markers_labeled, current_mask are xp arrays (CuPy if GPU_ENABLED)
           
            if xp.__name__ == 'cupy':
                print("DEBUG: GPU path for watershed conversion")
                # convert CuPy arrays to NumPy arrays
                image_for_watershed_np = xp.asnumpy(-distance)
                markers_for_watershed_np = xp.asnumpy(markers_labeled)
                mask_for_watershed_np = xp.asnumpy(current_mask)
            else: # xp is numpy
                print("DEBUG: CPU path for watershed conversion")
                image_for_watershed_np = -distance
                markers_for_watershed_np = markers_labeled
                mask_for_watershed_np = current_mask
            # Debug prints to verify types RIGHT BEFORE the watershed call
            print(f"DEBUG: About to call watershed. Types are:")
           
            connectivity_val = 2 
          
            # Call watershed with the (now hopefully) NumPy arrays
            watershed_segments_np = watershed(
                image_for_watershed_np,
                markers=markers_for_watershed_np,
                connectivity=connectivity_val,
                mask=mask_for_watershed_np
            )
        
            watershed_segments = xp.asarray(watershed_segments_np) # Convert result back to xp array
            # Now you can proceed with watershed_segments (which is an xp.array)
            # unique_segment_labels = xp.unique(watershed_segments) # If xp.unique is preferred
            # or
            unique_segment_labels = xp.unique(watershed_segments)
            num_actual_segments_created = 0
            for seg_label in unique_segment_labels:
                if seg_label == 0:  # This is the watershed line or background
                    continue
                segment_mask = (watershed_segments == seg_label).astype(xp.uint8)
                if xp.sum(segment_mask) > 0: # Ensure the segment is not empty
                    processing_queue.append(segment_mask) # Add new segments back to queue for size check
                    num_actual_segments_created +=1
            
            if num_actual_segments_created <= 1 and current_area > max_blob_area:
                # If watershed didn't split the region or resulted in one segment
                # that is essentially the original large region, add the original back
                # to avoid losing it or infinite loops.
                output_blob_masks.append(current_mask)

    # Final filter for any empty masks, though prior checks should mostly prevent this.
    print(f"Final masks ({len(output_blob_masks)})")
    final_masks = [mask for mask in output_blob_masks if xp.any(mask)]
    print(f"Final masks ({len(final_masks)})")
    
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
    """Checks if two bounding boxes overlap, with an optional margin."""
    # box format: (x1, y1, x2, y2)
    return not (box1[2] + margin < box2[0] or  # box1 is to the left of box2
                box1[0] - margin > box2[2] or  # box1 is to the right of box2
                box1[3] + margin < box2[1] or  # box1 is above box2
                box1[1] - margin > box2[3])   # box1 is below box2

def get_font_for_label(label_data: dict, font_path_str: Optional[str], default_font_size_from_cli: int) -> ImageFont.FreeTypeFont:
    """Loads a font object for a given label's data."""
    font_sz = label_data.get("font_size", default_font_size_from_cli)
    font_to_use = None
    if font_path_str and os.path.isfile(font_path_str):
        try:
            font_to_use = ImageFont.truetype(font_path_str, font_sz)
        except IOError:
            pass # Font file issue, will fall back
    
    if font_to_use is None: # Fallback to default system font
        try:
            # Attempt to load a specific common font if truetype fails or no path
            # This is just an example, you might prefer ImageFont.load_default() directly
            font_to_use = ImageFont.truetype("arial.ttf", font_sz) 
        except IOError:
            try: # Try Pillow's load_default with size if available
                font_to_use = ImageFont.load_default(size=font_sz)
            except TypeError: # Older Pillow might not support size argument
                font_to_use = ImageFont.load_default()
            except AttributeError: # In case load_default itself has issues
                 font_to_use = ImageFont.load_default() # Simplest fallback
    return font_to_use

def calculate_label_screen_bbox(label_data: dict, font_object: ImageFont.FreeTypeFont,
                                dummy_draw_context: ImageDraw.ImageDraw,
                                additional_nudge_pixels_up: float = 0.0) -> tuple[float, float, float, float]:
    """Calculates the screen bounding box of a label."""
    lx, ly = label_data["position"]
    text_val = str(label_data["value"])
    
    # Pillow 9.2.0+ uses getbbox, older versions textbbox with (0,0) offset
    try:
        # Using getbbox if available (more modern Pillow)
        bbox_at_origin = font_object.getbbox(text_val)
        # getbbox returns (left, top, right, bottom) relative to origin (0,0)
        # For text anchored at (0,0) top-left, left and top are often 0 or small negative values (descenders)
        # We need width and height based on this.
        text_w = bbox_at_origin[2] - bbox_at_origin[0]
        text_h = bbox_at_origin[3] - bbox_at_origin[1]
        # Adjust for anchor="mm" behavior: position is center.
        # Pillow's draw.text(anchor="mm") centers the *entire* bbox including ascenders/descenders.
        # The bbox from getbbox is what text() would use.
        # So, if (lx,ly) is the center, the top-left (x1,y1) for the bbox is:
        # x1 = lx - text_w / 2
        # y1 = ly - text_h / 2 (then adjust for nudge)
    except AttributeError: # Fallback for older Pillow using textbbox
        try:
            bbox_pil = dummy_draw_context.textbbox((0, 0), text_val, font=font_object)
            text_w = bbox_pil[2] - bbox_pil[0]
            text_h = bbox_pil[3] - bbox_pil[1]
        except AttributeError: # Further fallback if textbbox also fails (e.g. basic font)
            font_sz = label_data.get("font_size", 10) # Default to 10 if not found
            text_w = font_sz * len(text_val) * 0.6 # Rough estimate
            text_h = font_sz # Rough estimate

    effective_y_center = float(ly) - additional_nudge_pixels_up
    
    # Assuming (lx, ly) from label_data["position"] is intended to be the *center* of the text.
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
    """
    Resolves label collisions by prioritizing labels based on a strategy.
    Optimized to use xp (CuPy/NumPy) for neighbor calculations and sorting.
    Includes a workaround for cupy.lexsort with tuple inputs in older versions.
    """
    all_labels_initial_data = [] 
    dummy_image = Image.new("RGB", (1, 1)) 
    draw_context_for_bbox_calc = ImageDraw.Draw(dummy_image)

    label_id_counter = 0
    # ... (rest of the initial data collection loop as in segment_py_optimized_collision) ...
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

    if not all_labels_initial_data:
        return primitives_list

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

    # --- Vectorized Neighbor Count Calculation (as before) ---
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

    # --- Sorting ---
    # Define sort keys based on strategy.
    # Workaround for cupy.lexsort tuple issue:
    # If GPU_ENABLED (xp is cupy), pass a 2D array created by xp.stack.
    # The order of arrays in xp.stack should be (primary_key, secondary_key, ...).
    # If not GPU_ENABLED (xp is numpy), pass a tuple (..., secondary_key, primary_key).

    keys_for_lexsort: Union[tuple[xp.ndarray, ...], xp.ndarray]

    if strategy == "smallest_area_only":
        sorted_indices = xp.argsort(xp_region_areas)
    elif strategy == "fewest_neighbors_only":
        sorted_indices = xp.argsort(xp_neighbor_counts)
    elif strategy == "smallest_area_then_neighbors":
        # Primary key: xp_region_areas, Secondary key: xp_neighbor_counts
        if GPU_ENABLED: # CuPy workaround/preference for stacked array
            keys_for_lexsort = xp.stack((xp_region_areas, xp_neighbor_counts), axis=0)
        else: # NumPy standard tuple format
            keys_for_lexsort = (xp_neighbor_counts, xp_region_areas)
        sorted_indices = xp.lexsort(keys_for_lexsort)
    else:  # Default: "fewest_neighbors_then_area"
        # Primary key: xp_neighbor_counts, Secondary key: xp_region_areas
        if GPU_ENABLED: # CuPy workaround/preference for stacked array
            keys_for_lexsort = xp.stack((xp_neighbor_counts, xp_region_areas), axis=0) # This line corresponds to the traceback
        else: # NumPy standard tuple format
            keys_for_lexsort = (xp_region_areas, xp_neighbor_counts)
        sorted_indices = xp.lexsort(keys_for_lexsort)


    # --- Place Labels and Check Overlaps (as before) ---
    final_kept_label_ids = set()
    placed_bboxes_tuples = [] 

    if GPU_ENABLED:
        sorted_indices_np = xp.asnumpy(sorted_indices)
    else:
        sorted_indices_np = sorted_indices

    for i in sorted_indices_np:
        candidate_bbox_coords = tuple(xp_bboxes[i].tolist())
        has_collision = False
        for placed_bbox_tuple in placed_bboxes_tuples:
            if do_bboxes_overlap(candidate_bbox_coords, placed_bbox_tuple, margin=1):
                has_collision = True
                break
        if not has_collision:
            final_kept_label_ids.add(int(xp_ids[i].item()) if hasattr(xp_ids[i], 'item') else int(xp_ids[i]))
            placed_bboxes_tuples.append(candidate_bbox_coords)
            
    # --- Update Primitives List (as before) ---
    for primitive_data in primitives_list:
        kept_labels_for_this_primitive = []
        for label_obj_in_primitive in primitive_data["labels"]:
            if label_obj_in_primitive.get("id") in final_kept_label_ids:
                kept_labels_for_this_primitive.append(label_obj_in_primitive)
        primitive_data["labels"] = kept_labels_for_this_primitive
        
    return primitives_list

def find_stable_label_pixel(region_mask: xp.ndarray) -> tuple[int, int]:
    """
    Finds a 'stable' pixel within the True regions of the input mask.
    Optimized for GPU (CuPy) if available, otherwise uses NumPy.
    """
    h, w = region_mask.shape
    if xp.sum(region_mask) == 0:
        return (w // 2, h // 2)

    def _scan_runs_rowwise_left_to_right(matrix_rows: xp.ndarray) -> xp.ndarray:
        """
        Calculates run lengths of 1s row-wise, scanning from right to left.
        Uses a Numba CUDA kernel if GPU_ENABLED and xp is CuPy.
        Otherwise, falls back to a loop suitable for NumPy.
        """
        num_rows, num_cols = matrix_rows.shape
        # Ensure counts array is of a type Numba kernel can write to (e.g., int32)
        # and matches the device of matrix_rows.
        counts = xp.zeros_like(matrix_rows, dtype=xp.int32) 

        if num_cols == 0:
            return counts

        if GPU_ENABLED and xp.__name__ == 'cupy':
            # Ensure input matrix is suitable for the kernel (e.g., uint8 for 0s and 1s)
            # The kernel expects 0s and 1s. If matrix_rows is boolean, convert.
            matrix_for_kernel = matrix_rows.astype(xp.uint8) if matrix_rows.dtype == bool else matrix_rows

            # Define CUDA kernel launch parameters
            threads_per_block = 256  # Common choice, can be tuned (e.g., 128, 256, 512)
            # Calculate the number of blocks needed to cover all rows
            blocks_per_grid = (num_rows + (threads_per_block - 1)) // threads_per_block
        
            print(f"  Launching Numba CUDA kernel for scan: {blocks_per_grid} blocks, {threads_per_block} threads/block")
            scan_rows_kernel_for_gpu[blocks_per_grid, threads_per_block](
                matrix_for_kernel, counts, num_cols
            )
            # Kernel launch is asynchronous by default. CuPy operations
            # on 'counts' afterwards will implicitly synchronize, or you can explicitly sync.
        else:
            # Fallback for NumPy (or if GPU is not enabled/xp is not CuPy)
            # This is the original loop which is efficient for NumPy.
            # Ensure matrix_rows is int (0 or 1) for the multiplication trick.
            matrix_int = matrix_rows.astype(int) # Ensure 0/1 for multiplication
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
    if not contour or len(contour) < 2 : # Need at least two points to interpolate
        return contour # Return original if not enough points or empty

    for i in range(len(contour) - 1):
        x0, y0 = contour[i]
        x1, y1 = contour[i+1]

        dist = np.hypot(x1 - x0, y1 - y0) # Using np.hypot for scalar math to avoid transfer between CPU and GPU
        num_steps = max(1, int(dist / step))

        for j in range(num_steps): # Iterate up to num_steps - 1 to avoid duplicating next point
            t = j / float(num_steps)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            dense.append((x, y))
    dense.append(contour[-1]) # Ensure the very last point of the original contour is added
    return dense


def render_raster_from_primitives(canvas_size: tuple[int,int], primitives: list[dict],
                                  font_path: Optional[str] = None,
                                  additional_nudge_pixels_up: float = 0.0) -> Image.Image:
    width, height = canvas_size
    output_img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(output_img)
    pbn_color = (102, 204, 255) # Default PBN outline/text color

    font_path_str = str(font_path) if font_path else None

    for region_primitive in primitives:
        for contour_points_list in region_primitive.get("outline", []):
            if len(contour_points_list) > 1:
                 draw.line(contour_points_list, fill=pbn_color, width=1)
            elif contour_points_list: # Single point
                 draw.point(contour_points_list[0], fill=pbn_color)

        for label_data in region_primitive["labels"]:
            # Use a default font size from label_data if available, else a fallback.
            # default_font_size_for_fallback is not directly available here,
            # so use label_data's font_size or a hardcoded default.
            default_render_font_size = label_data.get("font_size", 10) 
            font_to_use = get_font_for_label(label_data, font_path_str, default_render_font_size)
            
            lx, ly = label_data["position"]
            text_value = str(label_data["value"])
            
            # Adjust y for rendering based on nudge (consistent with bbox calculation)
            effective_y_for_anchor = float(ly) - additional_nudge_pixels_up

            try:
                # Using anchor="mm" centers the text at (lx, effective_y_for_anchor)
                draw.text((float(lx), effective_y_for_anchor), text_value,
                          font=font_to_use, fill=pbn_color, anchor="mm")
            except (TypeError, AttributeError, ValueError) as e:
                # Fallback if anchor="mm" fails or other text issues
                # Calculate bbox again for manual top-left placement (less ideal)
                # This requires a dummy draw context if not available.
                # For simplicity, this fallback might be rough.
                # print(f"Warning: Fallback text rendering for label '{text_value}'. Error: {e}")
                # Fallback: draw at (lx, ly) as top-left, which might not be centered.
                draw.text((float(lx), float(ly)), text_value, font=font_to_use, fill=pbn_color)
                          
    return output_img

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
        
        current_region_mask = xp.array(mask_img, dtype=xp.uint8)
        if not xp.any(current_region_mask): continue # Skip if mask is empty

        blobs = blobbify_region(current_region_mask, min_blob_area, max_blob_area)
        for blob_mask_array in blobs: # Renamed 'blob_mask'
            if not xp.any(blob_mask_array): continue

            # 1. Prepare input for sklabel (expects NumPy)
            if xp.__name__ == 'cupy':
                blob_mask_for_sklabel_np = xp.asnumpy(blob_mask_array)
            else:
                blob_mask_for_sklabel_np = blob_mask_array # It's already NumPy

            sub_labeled_array_np = sklabel(blob_mask_for_sklabel_np, connectivity=1)

            props_for_loop = regionprops(sub_labeled_array_np)

            for sub_region_props in props_for_loop:
                area = sub_region_props.area
                # Create a mask for the current sub_region_props relative to sub_labeled_array
                current_sub_mask_np_comparison = (sub_labeled_array_np == sub_region_props.label)
                current_sub_mask = xp.asarray(current_sub_mask_np_comparison, dtype=xp.uint8)

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
    for blob_info in processed_masks_for_blobs: 
        print(f"DEBUG: {blob_info['region_id']} of area {blob_info['area']}")
        if blob_info["area"] >= min_blob_area:
            final_kept_blobs.append(blob_info)
            used_blob_ids.add(blob_info["region_id"]) 
        else:
            temp_small_blobs.append(blob_info)
            
    # Second pass: try to merge small blobs
    print("DEBUG: second blob merge pass...")
    for small_blob_info in temp_small_blobs:
        if small_blob_info["region_id"] in used_blob_ids: # Already merged or kept
            continue

        current_mask = small_blob_info["mask"]
        dilated_mask = ndi_xp.binary_dilation(current_mask, iterations=1) # Ensure ndi is imported
        
        best_match_target = None
        # Prefer merging with already kept large blobs of the same color
        # Then other large blobs, then other small blobs of same color.
        
        # Check against final_kept_blobs (larger ones) first
        potential_merge_targets = []
        for target_blob_info in final_kept_blobs + [b for b in temp_small_blobs if b["region_id"] != small_blob_info["region_id"] and b["region_id"] not in used_blob_ids]:
            if target_blob_info["region_id"] == small_blob_info["region_id"]: continue # Don't merge with self
            if xp.any(dilated_mask & target_blob_info["mask"]): # If they overlap/touch
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
                target_to_update["mask"] = ((target_to_update["mask"] | current_mask) > 0).astype(xp.uint8)
                target_to_update["area"] = target_to_update["mask"].sum()
                used_blob_ids.add(small_blob_info["region_id"]) # Mark small blob as merged
            else: # Should not happen if best_match_target was found
                final_kept_blobs.append(small_blob_info) # Failsafe: keep it if target not found for update
                used_blob_ids.add(small_blob_info["region_id"])
            print("DEBUG: Found best match to merge...")
        else: # No neighbor to merge with, keep it if it wasn't used.
            print("DEBUG: No neighbor to merge with...")
            if small_blob_info["region_id"] not in used_blob_ids:
                 final_kept_blobs.append(small_blob_info)
                 used_blob_ids.add(small_blob_info["region_id"])
    
    # Filter out any blobs that were merged into others and might still be in final_kept_blobs by mistake
    final_blobs_for_primitives = [b for b in final_kept_blobs if b["mask"].sum() >= min_blob_area]


    for blob_data in final_blobs_for_primitives: # Renamed 'm' to 'blob_data'
        current_blob_mask = blob_data["mask"]
        if not xp.any(current_blob_mask): continue

        if xp.__name__ == 'cupy':
            current_blob_mask_np = xp.asnumpy(current_blob_mask)
        else: # xp is numpy
            current_blob_mask_np = current_blob_mask

        # Call find_contours with the NumPy array
        print("DEBUG: blobbify_primitives: finding contours...")
        contours_list_np = find_contours(current_blob_mask_np, level=0.5) 

        if not contours_list_np: # Check if the list of contours is empty
            continue

        dense_outlines = []
        for contour_path in contours_list_np: # Renamed 'contour' to 'contour_path'

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
        label_strategy="diagonal",
        small_region_label_strategy="stable", # For small region fallback strategy
        additional_nudge_pixels_up=0, # For consistent label rendering and bbox calculation
        interpolate_contours=True
        ):
    image = Image.open(input_path).convert("RGB")
    img_data = xp.array(image) # img_data is now on GPU if xp is cupy
    height, width = img_data.shape[:2]
    primitives = []
    region_id_counter = 0

    # Create dummy draw object ONCE for all bbox calculations within this function
    dummy_img_for_bbox_calc = Image.new("L", (1,1))
    dummy_draw_context = ImageDraw.Draw(dummy_img_for_bbox_calc)

    font_path_str = str(font_path) if font_path else None # Convert Path to str for os.path.isfile

    # --- Dynamic Minimum Area Calculation (existing logic) ---
    base_font_size_for_calc = font_size if font_size is not None else 12
    # ... (rest of your dynamic_min_area_for_font calculation logic remains unchanged) ...
    # Ensure representative_text and font_for_measurement are defined as in your original code
    representative_text = "8" # Simplified, ensure this matches your original logic for accuracy
    if palette is not None and len(palette) > 0:
        max_idx = len(palette) -1
        if max_idx < 10: representative_text = "8"
        elif max_idx < 100: representative_text = "88"
        else: representative_text = "888"

    # Ensure font_for_measurement is loaded as in your original code
    font_for_measurement = None
    if font_path_str and os.path.isfile(font_path_str):
        try: font_for_measurement = ImageFont.truetype(font_path_str, base_font_size_for_calc)
        except IOError: pass
    if font_for_measurement is None:
        try: font_for_measurement = ImageFont.load_default(size=base_font_size_for_calc)
        except TypeError: font_for_measurement = ImageFont.load_default()
        # except AttributeError: font_for_measurement = ImageFont.load_default() # PIL Deprecation

    try:
        # Pillow 9.2.0+ uses getbbox, older versions textbbox with (0,0) offset
        try:
            text_bbox_m = dummy_draw_context.getbbox((0,0), representative_text, font=font_for_measurement)
        except AttributeError: # Fallback for older Pillow
            text_bbox_m = dummy_draw_context.textbbox((0,0), representative_text, font=font_for_measurement)
        label_w_est = text_bbox_m[2] - text_bbox_m[0]
        label_h_est = text_bbox_m[3] - text_bbox_m[1]
    except (AttributeError, TypeError): # Further fallback if font issues
        label_w_est = base_font_size_for_calc * 0.6 * len(representative_text)
        label_h_est = base_font_size_for_calc

    label_w_est = max(1, label_w_est); label_h_est = max(1, label_h_est)
    dynamic_min_area_for_font = (label_w_est * label_h_est) * 2.5
    dynamic_min_area_for_font = max(dynamic_min_area_for_font, 25)
    actual_min_filter_area = max(min_region_area, int(dynamic_min_area_for_font))
    # typer.echo(...) for this info can be in pbngen.py

    # --- Iterate through palette colors and then regions ---
    for idx, color_rgb_val in enumerate(palette): # Renamed 'color' to 'color_rgb_val'
        color_val_on_device = xp.asarray(color_rgb_val, dtype=img_data.dtype) # Ensure dtype matches for comparison
        
        # Create mask for the current color on the device (GPU/CPU)
        mask = xp.all(img_data == color_val_on_device.reshape(1, 1, 3), axis=-1).astype(xp.uint8) # Reshape for broadcasting

        if not xp.any(mask): # Optimization: if mask is empty, skip labeling and regionprops
            continue

        # --- MODIFICATION START: Label connected regions using ndi_xp.label ---
        labeled_array_on_device: xp.ndarray
        num_features: int # To store the number of features found

        if GPU_ENABLED: # This global flag is defined at the top of segment.py
            # For cupyx.scipy.ndimage.label, structure defines connectivity.
            # For 4-connectivity (like skimage's connectivity=1 for 2D):
            # structure = xp.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
            # For 8-connectivity (like skimage's connectivity=2 for 2D, or default for scipy.ndimage.label):
            structure = xp.array([[1,1,1],[1,1,1],[1,1,1]], dtype=bool) # Using 8-connectivity as common default
            labeled_array_on_device, num_features = ndi_xp.label(mask, structure=structure)
        else: # xp is NumPy, ndi_xp is scipy.ndimage
            # For scipy.ndimage.label, connectivity=1 means 4-way, connectivity=2 means 8-way (default)
            # To match common default behavior or if 8-connectivity is preferred:
            labeled_array_on_device, num_features = ndi_xp.label(mask) # Default connectivity (8-way for 2D)
            # If 4-connectivity is strictly needed to match previous sklabel(connectivity=1):
            # labeled_array_on_device, num_features = ndi_xp.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool))


        if num_features == 0: # Optimization: if no regions found, skip regionprops
            continue
            
        # skimage.measure.regionprops expects a NumPy array for the label_image.
        # If labeled_array_on_device is on GPU (CuPy), convert it to CPU (NumPy).
        labeled_array_for_regionprops_np: np.ndarray
        if GPU_ENABLED:
            labeled_array_for_regionprops_np = xp.asnumpy(labeled_array_on_device)
        else:
            labeled_array_for_regionprops_np = labeled_array_on_device # It's already a NumPy array
        # --- MODIFICATION END ---

        # Pass the NumPy version of the labeled array to regionprops
        # regionprops itself runs on CPU.
        for region_props in regionprops(labeled_array_for_regionprops_np): # Existing loop
            if region_props.area < actual_min_filter_area:
                continue
            if region_props.area > 0.95 * (width * height) :
                continue

            # IMPORTANT: Use labeled_array_on_device for creating region_mask
            # This keeps region_mask on GPU if source was GPU, for subsequent GPU-accelerated functions.
            region_mask = (labeled_array_on_device == region_props.label).astype(xp.uint8)

            # The rest of your existing logic for processing each region_props:
            # (contour finding, label placement, elision, etc.)
            # The conversion of `region_mask` to NumPy for `find_contours`
            # using `xp.asnumpy(region_mask)` is already handled correctly later in your code.

            # region_mask is currently an xp.array (CuPy array if GPU_ENABLED is True)
            region_mask_for_contours_np = region_mask # Default to current type
            if GPU_ENABLED: # Or your GPU_ENABLED flag
                # Convert the CuPy array to a NumPy array on the CPU for find_contours
                region_mask_for_contours_np = xp.asnumpy(region_mask)
            
            contours = find_contours(region_mask_for_contours_np, level=0.5) # skimage function on NumPy array
            
            if not contours: continue

            minr, minc, maxr, maxc = region_props.bbox
            outlines = []
            for contour_path in contours:
                if not contour_path.size: continue
                flipped_path = [(x, y) for y, x in contour_path]
                dense_path = interpolate_contour(flipped_path, step=0.5) if interpolate_contours else flipped_path
                outline_coords = [(int(xf), int(yf)) for xf, yf in dense_path if 0 <= int(xf) < width and 0 <= int(yf) < height]
                if outline_coords and len(outline_coords) > 1:
                    outlines.append(outline_coords)

            if not outlines: continue

            local_font_size = font_size if font_size is not None else 12
            local_spacing_for_diagonal = tile_spacing or max(8, min(maxc - minc, maxr - minr) // 4)
            local_spacing_for_diagonal = max(1, local_spacing_for_diagonal)

            labels = []
            region_width = maxc - minc
            region_height = maxr - minr
            
            use_fallback_strategy = (region_width < local_spacing_for_diagonal or 
                                   region_height < local_spacing_for_diagonal)
            
            current_label_strategy = label_strategy
            if use_fallback_strategy:
                current_label_strategy = small_region_label_strategy

            if current_label_strategy == "stable":
                # find_stable_label_pixel is already optimized to use xp (GPU/CPU)
                sx_local, sy_local = find_stable_label_pixel(region_mask) # Uses xp array
                labels = [make_label(sx_local, sy_local, idx, local_font_size, region_area=region_props.area)]
            elif current_label_strategy == "centroid":
                cy_global, cx_global = int(region_props.centroid[0]), int(region_props.centroid[1])
                labels = [make_label(cx_global, cy_global, idx, local_font_size, region_area=region_props.area)]
            elif current_label_strategy == "diagonal":
                row_is_offset = False
                for y_coord in range(minr, maxr, local_spacing_for_diagonal):
                    current_x_offset = local_spacing_for_diagonal // 2 if row_is_offset else 0
                    for x_base_grid in range(minc, maxc, local_spacing_for_diagonal):
                        actual_x_coord = x_base_grid + current_x_offset
                        if not (0 <= actual_x_coord < width and 0 <= y_coord < height): continue
                        if region_mask[y_coord, actual_x_coord]: # Check on xp array
                            labels.append(make_label(actual_x_coord, y_coord, idx, local_font_size, region_area=region_props.area))
                    row_is_offset = not row_is_offset
            elif current_label_strategy == "none":
                labels = []

            # --- Elision Logic --- (Your existing elision logic)
            if len(labels) > 1:
                surviving_labels_for_this_region = []
                for label_candidate in labels:
                    font_obj_for_elision = get_font_for_label(label_candidate, font_path_str, local_font_size) # Use local_font_size
                    
                    l_bbox_x1, l_bbox_y1, l_bbox_x2, l_bbox_y2 = calculate_label_screen_bbox(
                        label_candidate, 
                        font_obj_for_elision, 
                        dummy_draw_context,
                        additional_nudge_pixels_up
                    )

                    pixels_in_bbox_total = 0
                    pixels_in_bbox_and_region_mask = 0
                    
                    scan_x_start = max(0, int(xp.floor(l_bbox_x1)))
                    scan_x_end = min(width, int(xp.ceil(l_bbox_x2)))
                    scan_y_start = max(0, int(xp.floor(l_bbox_y1)))
                    scan_y_end = min(height, int(xp.ceil(l_bbox_y2)))

                    for y_scan in range(scan_y_start, scan_y_end):
                        for x_scan in range(scan_x_start, scan_x_end):
                            if (l_bbox_x1 <= x_scan < l_bbox_x2 and
                                l_bbox_y1 <= y_scan < l_bbox_y2):
                                pixels_in_bbox_total += 1
                                if region_mask[y_scan, x_scan]: # Check on xp array
                                    pixels_in_bbox_and_region_mask += 1
                    
                    percentage_outside = 100.0
                    if pixels_in_bbox_total > 0:
                        percentage_outside = (1.0 - (pixels_in_bbox_and_region_mask / float(pixels_in_bbox_total))) * 100.0
                    
                    if percentage_outside <= 25.0:
                        surviving_labels_for_this_region.append(label_candidate)
                labels = surviving_labels_for_this_region
            
            if outlines and labels:
                 primitives.append(dict(
                    outline=outlines,
                    labels=labels,
                    region_id=region_id_counter,
                    color=tuple(int(c) for c in color_rgb_val),
                    palette_index=idx,
                ))
            region_id_counter += 1

    return primitives
