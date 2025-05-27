from PIL import Image, ImageFilter, ImageDraw
import random
import os # Added for robust path handling if needed, though not strictly used in this version.

def apply_style(input_path, output_path, style, blur_radius=None, pixelate_block_size=None, mosaic_block_size=None, edge_strength=None):
    """
    Applies a specified visual style to an input image and saves the output.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the styled image.
        style (str): The name of the style to apply (e.g., "blur", "pixelate", "mosaic", "impressionist").
        blur_radius (int, optional): Radius for Gaussian blur, used by "blur" and "impressionist" styles.
        pixelate_block_size (int, optional): Block size for "pixelate" style.
        mosaic_block_size (int, optional): Block size for "mosaic" style.

    Raises:
        ValueError: If the input file is not found, cannot be opened, or if an unknown style is specified.
    """
    try:
        image = Image.open(input_path)
    except FileNotFoundError:
        raise ValueError(f"Error: Input file not found at {input_path}")
    except Exception as e:
        # Catch other PIL-related errors during open
        raise ValueError(f"Error opening image {input_path}: {e}")

    if style == "blur":
        # Determine the actual blur radius to use.
        # If blur_radius is not provided by the caller, default to 4.
        actual_blur_radius = blur_radius if blur_radius is not None else 4
        styled_image = image.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
        styled_image.save(output_path)

    elif style == "pixelate":
        # Placeholder for your pixelation logic
        # Example:
        # actual_block_size = pixelate_block_size if pixelate_block_size is not None else 10
        # img_small = image.resize((image.width // actual_block_size, image.height // actual_block_size), Image.Resampling.BILINEAR)
        # styled_image = img_small.resize(image.size, Image.Resampling.NEAREST)
        # styled_image.save(output_path)
        print(f"Stylize: 'pixelate' style with block_size {pixelate_block_size} (Not fully implemented in this example). Saving original.")
        image.save(output_path) # Fallback: save original if not implemented

    elif style == "mosaic":
        # Placeholder for your mosaic logic
        print(f"Stylize: 'mosaic' style with block_size {mosaic_block_size} (Not fully implemented in this example). Saving original.")
        image.save(output_path) # Fallback: save original if not implemented

    elif style == "impressionist":
        w, h = image.size
        
        # Ensure the image is in RGB mode for consistent color handling
        image_rgb = image.convert("RGB")

        # Determine the blur radius for the source of brush colors.
        # If blur_radius is not specified by the user for this style, use a default of 10.
        actual_blur_radius = blur_radius
        if actual_blur_radius is None:
            actual_blur_radius = 10 
        
        blurred_source = image_rgb.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
        
        # Create a new canvas to paint the impressionistic strokes on.
        # Starting with a white background.
        output_canvas = Image.new("RGB", (w, h), (255, 255, 255)) 
        draw = ImageDraw.Draw(output_canvas)

        # Define brush diameters. These are inspired by your 512x512 example [8, 4, 2].
        # For more general applicability, these could be scaled relative to image size
        # or made configurable.
        # Example of dynamic scaling:
        # base_dim = min(w,h)
        # brush_diameters = [max(3, int(base_dim * 0.03)), max(2, int(base_dim * 0.015)), max(1, int(base_dim * 0.0075))]
        brush_diameters_to_use = [16, 8, 4] # Using fixed sizes as per your example

        for brush_d in brush_diameters_to_use:
            if brush_d < 1: # Ensure brush diameter is practical
                continue

            paint_locations = []
            # Determine the step for placing brush stroke centers.
            # A step of brush_d / 2 provides 50% overlap.
            #step = max(1, brush_d // 2) 
            step = 4

            # Create a grid of base points for brush strokes
            for y_base in range(0, h, step):
                for x_base in range(0, w, step):
                    # Add random jitter to the brush center for a more organic, less grid-like appearance
                    jitter_amount = step // 2 # Jitter up to half the step size
                    jitter_x = random.randint(-jitter_amount, jitter_amount)
                    jitter_y = random.randint(-jitter_amount, jitter_amount)
                    jitter_x = 0
                    jitter_y = 0 
                    # Calculate the final center position for the brush stroke
                    # Ensure the point stays within the image boundaries for color sampling
                    center_x = max(0, min(w - 1, x_base + jitter_x))
                    center_y = max(0, min(h - 1, y_base + jitter_y))
                    paint_locations.append((center_x, center_y))
            
            # Shuffle the locations to apply strokes in a random order for the current brush size
            random.shuffle(paint_locations)

            for px, py in paint_locations:
                # Sample the color from the blurred source image at the brush's center point
                try:
                    color = blurred_source.getpixel((px, py))
                except IndexError:
                    # This should ideally not be reached if center_x, center_y are clamped correctly
                    color = (0,0,0) # Fallback color if somehow out of bounds
                
                # Define the bounding box for the circular brush (ellipse)
                # (px, py) is the center of the stroke.
                half_brush_d = brush_d / 2.0 # Use float for centering, PIL handles conversion.
                x0 = px - half_brush_d
                y0 = py - half_brush_d
                x1 = px + half_brush_d # Bounding box extends from center - radius to center + radius
                y1 = py + half_brush_d
                
                # Draw the brush stroke
                draw.ellipse([(x0, y0), (x1, y1)], fill=color)
                
        styled_image = output_canvas # The final image is the canvas with all strokes
        styled_image.save(output_path)

    else:
        raise ValueError(f"Unknown style: {style}. Supported styles: blur, pixelate, mosaic, impressionist.")

if __name__ == '__main__':
    # Example usage (creates dummy files for testing)
    print("Running stylize.py example...")
    # Create a dummy input image
    if not os.path.exists("test_outputs"):
        os.makedirs("test_outputs")
    
    try:
        img = Image.new("RGB", (256, 256), color = (150, 120, 200))
        draw_test = ImageDraw.Draw(img)
        draw_test.rectangle([(50,50), (150,150)], fill=(200,50,50))
        draw_test.ellipse([(100,100), (200,200)], fill=(50,200,50))
        img.save("dummy_input.png")

        print("Applying blur style...")
        apply_style("dummy_input.png", "test_outputs/styled_blur.png", "blur", blur_radius=5)
        print("Applying impressionist style...")
        apply_style("dummy_input.png", "test_outputs/styled_impressionist.png", "impressionist", blur_radius=8) # Example blur_radius for impressionist
        print("Applying impressionist style (default blur)...")
        apply_style("dummy_input.png", "test_outputs/styled_impressionist_default_blur.png", "impressionist")
        
        print(f"Example styled images saved in '{os.path.abspath('test_outputs')}' directory.")
        print(f"Dummy input 'dummy_input.png' created in current directory: {os.path.abspath('.')}")

    except ImportError:
        print("Pillow (PIL) library is not installed. This example requires Pillow.")
    except Exception as e:
        print(f"An error occurred during the example: {e}")

