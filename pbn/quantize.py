from PIL import Image, ImageEnhance # ImageEnhance not used here, but kept if other functions might use it
import numpy as np
from sklearn.cluster import KMeans
from pbn.palette_tools import map_image_to_palette # Assuming this is from your pbn package
import traceback # Kept, though not explicitly used in this snippet
from typing import Tuple, Optional # For type hinting


def quantize_image(
    input_path: str, 
    num_colors: Optional[int] = None, 
    fixed_palette: Optional[np.ndarray] = None, 
    dither: Optional[bool] = None, # Changed from PIL.Image.Dither to bool for clarity
    sort_by_frequency: bool = True
) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """
    Quantize image using either KMeans or a fixed palette.
    The returned palette can be sorted by color frequency in the image.
    The function returns the quantized PIL Image object and the palette.

    Args:
        input_path (str): Path to the input image file.
        num_colors (int, optional): Number of colors for KMeans. Ignored if fixed_palette is used.
                                    Defaults to 12 if None and fixed_palette is not provided.
        fixed_palette (np.ndarray, optional): Optional pre-extracted RGB palette (shape [N, 3]).
        dither (bool, optional): If True, applies Floyd-Steinberg dithering during initial RGB conversion
                                 if adaptive palette is used (i.e., fixed_palette is None).
                                 Note: Dithering here is basic. For more control, pre-process image.
        sort_by_frequency (bool): If True, sorts the returned palette by color frequency (most frequent first).

    Returns:
        Tuple[Optional[PIL.Image.Image], Optional[np.ndarray]]:
            - The quantized PIL Image object (RGB mode).
            - Array of RGB palette colors (np.ndarray, shape [M, 3], dtype=np.uint8),
              sorted by frequency if requested.
            Returns (None, None) on error.
    """
    try:
        if dither and fixed_palette is None and num_colors is not None:
            # Apply dithering during an initial quantization pass if no fixed palette.
            # This is a simple way to achieve dithering; might not be optimal for all cases.
            # Image.ADAPTIVE with colors specified will generate its own palette.
            image = Image.open(input_path).convert('RGB') # Ensure RGB before this kind of quantize
            image = image.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            image = image.convert("RGB") # Convert back to RGB after palette quantization
        else:
            image = Image.open(input_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None, None
    except Exception as e:
        print(f"Error opening image {input_path}: {e}")
        return None, None
    
    img_data_np = np.array(image) # Original or dither-quantized image pixels as NumPy array

    quantized_rgb_array: np.ndarray 
    palette_to_analyze: np.ndarray

    if fixed_palette is not None:
        if not isinstance(fixed_palette, np.ndarray) or fixed_palette.ndim != 2 or fixed_palette.shape[1] != 3:
            print("Error: fixed_palette must be a NumPy array of shape [N, 3].")
            return None, None
        # Ensure fixed_palette is uint8 for consistency with KMeans output and image data
        fixed_palette = fixed_palette.astype(np.uint8)
        # map_image_to_palette returns an image with pixels mapped to colors from fixed_palette
        quantized_rgb_array = map_image_to_palette(img_data_np, fixed_palette) # Ensure this handles RGB
        palette_to_analyze = fixed_palette
    else: 
        if num_colors is None: 
            num_colors = 12 
        if not (1 <= num_colors <= 256) : # Practical limits for KMeans here
            print(f"Warning: num_colors ({num_colors}) for KMeans is out of typical range (1-256). Clamping or check usage.")
            num_colors = max(1, min(num_colors, 256))

        img_array_reshaped = img_data_np.reshape((-1, 3))
        try:
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto') 
            labels = kmeans.fit_predict(img_array_reshaped)
            palette_from_kmeans = kmeans.cluster_centers_.astype(np.uint8) # Ensure uint8
        except Exception as e:
            print(f"Error during KMeans clustering: {e}")
            return None, None
        
        quantized_rgb_array = palette_from_kmeans[labels].reshape(img_data_np.shape)
        palette_to_analyze = palette_from_kmeans

    final_palette_to_return: np.ndarray

    if sort_by_frequency:
        unique_colors_in_image, counts_for_unique_colors = np.unique(
            quantized_rgb_array.reshape(-1, 3), axis=0, return_counts=True
        )
        # Create a map of actual colors found in the image to their counts
        color_to_actual_count_map = {
            tuple(color): count for color, count in zip(unique_colors_in_image, counts_for_unique_colors)
        }
        
        palette_with_counts = []
        # Iterate through the colors in `palette_to_analyze` (either from K-Means or fixed_palette)
        # and get their counts from the map.
        for color_in_palette in palette_to_analyze:
            count = color_to_actual_count_map.get(tuple(color_in_palette), 0)
            palette_with_counts.append({'color_rgb': color_in_palette, 'count': count})
        
        sorted_palette_entries = sorted(palette_with_counts, key=lambda x: x['count'], reverse=True)
        
        # Filter out colors from palette_to_analyze that didn't appear in the image *if* they have a count of 0
        # This is important if fixed_palette contained colors not present in the quantized image.
        # For KMeans, all its cluster centers *should* ideally be represented, but good to be robust.
        final_palette_to_return = np.array([entry['color_rgb'] for entry in sorted_palette_entries if entry['count'] > 0 or not fixed_palette], dtype=np.uint8)
        if len(final_palette_to_return) == 0 and len(sorted_palette_entries) > 0: # Handle case where all counts are 0 (e.g. single color image with fixed_palette mismatch)
            final_palette_to_return = np.array([sorted_palette_entries[0]['color_rgb']], dtype=np.uint8) # Return at least one color
        elif len(final_palette_to_return) == 0 and len(palette_to_analyze) > 0 : # If everything was filtered, fall back to original palette_to_analyze
             final_palette_to_return = palette_to_analyze.astype(np.uint8)


    else: # Use the palette in the order it came
        final_palette_to_return = palette_to_analyze.astype(np.uint8) 
    
    # Create the PIL Image from the RGB quantized array
    quantized_pil_image_output = Image.fromarray(quantized_rgb_array.astype(np.uint8), 'RGB') # Ensure RGB mode
    
    # The function no longer saves the file or uses pnginfo.
    # It returns the PIL image and the palette.
    return quantized_pil_image_output, final_palette_to_return