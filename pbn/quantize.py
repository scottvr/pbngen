from PIL import Image, ImageEnhance
import numpy as np
from sklearn.cluster import KMeans
from pbn.palette_tools import map_image_to_palette
import traceback





def quantize_image(input_path, output_path, num_colors=None, fixed_palette=None, pnginfo=None, dither=None):
    """
    Quantize image using either KMeans or a fixed palette.
    The returned palette is sorted by color frequency in the image (most frequent first).
    The saved output_path PNG contains the quantized RGB image.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the quantized output image.
        num_colors (int): Number of colors for KMeans (ignored if fixed_palette is used).
        fixed_palette (np.ndarray): Optional pre-extracted RGB palette.
        pnginfo (PIL.PngImagePlugin.PngInfo, optional): PngInfo object for metadata.


    Returns:
        np.ndarray: Array of RGB palette colors, sorted by frequency (most frequent first).
    """
    if dither is not None:
        image = Image.open(input_path).convert('RGB', dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE, colors=num_colors)
    else:
        image = Image.open(input_path).convert("RGB")
    
    img_data_np = np.array(image) # Original image pixels as NumPy array

    # This will hold the RGB version of the quantized image
    quantized_rgb_array: np.ndarray 
    # This will hold the palette whose colors need to be frequency-sorted
    palette_to_analyze: np.ndarray

    if fixed_palette is not None:
        # map_image_to_palette returns an image with pixels mapped to colors from fixed_palette
        quantized_rgb_array = map_image_to_palette(img_data_np, fixed_palette)
        palette_to_analyze = fixed_palette
    else: # Else use KMeans to determine palette and quantize
        if num_colors is None: # Default num_colors if not provided for KMeans
            num_colors = 12 # A sensible default
        img_array_reshaped = img_data_np.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto') # random_state cited
        labels = kmeans.fit_predict(img_array_reshaped)
        palette_from_kmeans = kmeans.cluster_centers_.astype("uint8")
        
        quantized_rgb_array = palette_from_kmeans[labels].reshape(img_data_np.shape)
        palette_to_analyze = palette_from_kmeans

    # Count frequencies of each color from palette_to_analyze in the quantized_rgb_array
    # 1. Get unique colors and their counts from the actual quantized image
    unique_colors_in_image, counts_for_unique_colors = np.unique(
        quantized_rgb_array.reshape(-1, 3), axis=0, return_counts=True
    )
    # 2. Create a map for quick lookup of counts for these unique colors
    color_to_actual_count_map = {
        tuple(color): count for color, count in zip(unique_colors_in_image, counts_for_unique_colors)
    }

    # 3. For each color in our target palette (palette_to_analyze), find its count
    palette_with_counts = []
    for i, color_in_palette in enumerate(palette_to_analyze):
        count = color_to_actual_count_map.get(tuple(color_in_palette), 0)
        palette_with_counts.append({
            'color_rgb': color_in_palette,    # Keep as np.array for easy reconstruction
            'original_index': i,              # In case it's ever needed
            'count': count
        })
    
    # 4. Sort these palette entries by count in descending order
    # Python's sort is stable, so original relative order of colors with same count is preserved.
    sorted_palette_entries = sorted(palette_with_counts, key=lambda x: x['count'], reverse=True)
    
    # 5. Construct the final sorted palette to be returned
    final_sorted_palette = np.array([entry['color_rgb'] for entry in sorted_palette_entries], dtype=np.uint8)

    # Save the quantized image (its pixel values are RGB based on the original quantization)
    quantized_pil_image_output = Image.fromarray(quantized_rgb_array)
    quantized_pil_image_output.save(output_path, pnginfo=pnginfo) # Pass pnginfo if provided

    return final_sorted_palette