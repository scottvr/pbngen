from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def extract_palette_from_image(path, max_colors=24):
    """
    Extract a fixed palette from an image (e.g. a paint tray).

    Args:
        path (str): Path to the palette image.
        max_colors (int): Maximum number of colors to extract.

    Returns:
        np.ndarray: Array of RGB colors (uint8) with shape (N, 3).
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((100, 100))  # Downsample for speed and uniformity
    pixels = np.array(image).reshape(-1, 3)

    # Run k-means to extract dominant swatch tones
    kmeans = KMeans(n_clusters=max_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    palette = kmeans.cluster_centers_.astype("uint8")

    return palette

def map_image_to_palette(image_array, palette):
    """
    Map every pixel in the image to the nearest color in the fixed palette.

    Args:
        image_array (np.ndarray): HxWx3 RGB image data
        palette (np.ndarray): Nx3 palette array

    Returns:
        np.ndarray: Quantized image array of same shape as input
    """
    h, w, _ = image_array.shape
    flat = image_array.reshape((-1, 3))

    # Find nearest palette color for each pixel
    dists = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    quantized_flat = palette[nearest]

    return quantized_flat.reshape((h, w, 3))
