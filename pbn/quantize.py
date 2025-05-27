from PIL import Image, ImageEnhance
import numpy as np
from sklearn.cluster import KMeans
from pbn.palette_tools import map_image_to_palette
import traceback

def quantize_image(input_path, output_path, num_colors=None, fixed_palette=None, dither=None):
    """
    Quantize image using either KMeans or a fixed palette.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the quantized output image.
        num_colors (int): Number of colors for KMeans (ignored if fixed_palette is used).
        fixed_palette (np.ndarray): Optional pre-extracted RGB palette.

    Returns:
        np.ndarray: Array of RGB palette colors used in the quantized image.
    """
    image = Image.open(input_path).convert("RGB")
    # increase contrast before processing
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast by 50%
    if dither:
        image = image.convert('RGB', dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE, colors=num_colors)
    img_data = np.array(image)

    if dither is not None:
        do_dither = True
    if fixed_palette is not None:
        quantized_array = map_image_to_palette(img_data, fixed_palette)
        quantized_img = Image.fromarray(quantized_array)
        quantized_img.save(output_path)
        return fixed_palette

    img_array = img_data.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    labels = kmeans.fit_predict(img_array)
    palette = kmeans.cluster_centers_.astype("uint8")

    quantized_array = palette[labels].reshape(img_data.shape)
    quantized_img = Image.fromarray(quantized_array)
    quantized_img.save(output_path)

    return palette
