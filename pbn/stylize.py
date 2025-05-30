from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageOps
import numpy as np
import random
import os 
import typer # for typer.echo
import math

def apply_style(input_path, output_path, style, blur_radius=None, pixelate_block_size=None, mosaic_block_size=None, brush_step=None, brush_size=None, edge_strength=None, num_brushes=None, fervor=None, focus=None):
    """
    Applies a specified visual style to an input image and saves the output.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the styled image.
        style (str): The name of the style to apply (e.g., "blur", "pixelate", "mosaic", "impressionist").
        blur_radius (int, optional): Radius for Gaussian blur, used by "blur" and "impressionist" styles.
        pixelate_block_size (int, optional): Block size for "pixelate" style.
        mosaic_block_size (int, optional): Block size for "mosaic" style.
        brush_size:
        focus:

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

    elif style == "smooth":
        styled_image = image.filter(ImageFilter.SMOOTH)
        styled_image.save(output_path)
        
    elif style == "smooth_more":
        styled_image = image.filter(ImageFilter.SMOOTH_MORE)
        styled_image.save(output_path)

    elif style == "pixelate":
        actual_block_size = pixelate_block_size if pixelate_block_size is not None else 10
        img_small = image.resize((image.width // actual_block_size, image.height // actual_block_size), Image.Resampling.BILINEAR)
        styled_image = img_small.resize(image.size, Image.Resampling.NEAREST)
        print(f"Stylize: 'pixelate' style with block_size {pixelate_block_size} (Not fully implemented in this example). Saving original.")
        styled_image.save(output_path)
#   elif style == "pixelate":
#        # Resize down and back up to simulate pixelation
#        w, h = image.size
#        if not pixelate_block_size: # user passed no argument
#            pixelate_block_size = 64
#        pixel_size = max(4, min(w, h) // pixelate_block_size)
#        styled = image.resize((w // pixel_size, h // pixel_size), resample=Image.NEAREST)
#        styled = styled.resize((w, h), resample=Image.NEAREST)

    elif style == "mosaic":
        print(f"Stylize: 'mosaic' style with block_size {mosaic_block_size} (Not fully implemented in this example). Saving original.")
        # Similar to pixelate but with bicubic upscale for blocky blending
        w, h = image.size
        if not mosaic_block_size:
            mosaic_block_size = 64
        block_size = max(4, min(w, h) // mosaic_block_size)
        styled_image = image.resize((w // block_size, h // block_size), resample=Image.NEAREST)
        styled_image = styled.resize((w, h), resample=Image.BICUBIC)
        styled_image.save(output_path)

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

        
    elif style=="test":
        w, h = image.size
        image_array = np.array(image)

        # temp hardcode:
        actual_num_brushes = num_brushes if num_brushes is not None else 2
        actual_num_brushes = min(actual_num_brushes, 5) # clamp to five brushes for now
        actual_brush_size = brush_size if brush_size is not None else 3
        actual_brush_step = brush_step if brush_step is not None else max(1, actual_brush_size // actual_num_brushes)
        typer.echo(f"")

        enhancer = ImageEnhance.Contrast(image)
        for idx in range(0, actual_num_brushes):
            bsize = actual_brush_size - idx * actual_brush_step
            actual_blur_radius = blur_radius if blur_radius is not None else 4
            focus = focus if focus is not None else 1
            image = Image.fromarray(image_array)
            image = enhancer.enhance(1.1)  # Increase contrast by 10%
            image = image.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
            typer.echo(f"Using #{bsize} brush ({idx+1}/{actual_num_brushes}).")
            for _ in range(w * h // (bsize**2) * int(focus)):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
    
                # Get the color of the pixel
                r, g, b = image.getpixel((x, y))
                # Apply brush strokes around the pixel
                for i in range(max(0, x - bsize // 2), min(w, x + bsize // 2 + 1)):
                    for j in range(max(0, y - bsize // 2), min(h, y + bsize // 2 + 1)):
                        dx = i - x
                        dy = j - y
                        if dx**2 + dy**2 <= (bsize // 2)**2:
                            # Calculate new color values with random variation
                            new_r_float = r + random.uniform(-20, 20)
                            new_g_float = g + random.uniform(-20, 20)
                            new_b_float = b + random.uniform(-20, 20)
    
                            # Clamp the values to the valid uint8 range [0, 255]
                            clamped_r = max(0, min(255, int(new_r_float)))
                            clamped_g = max(0, min(255, int(new_g_float)))
                            clamped_b = max(0, min(255, int(new_b_float)))
                            
                            # Assign the clamped color tuple
                            image_array[j, i] = (clamped_r, clamped_g, clamped_b)        
            typer.echo(f"{i*j} strokes.")
        styled_image = Image.fromarray(image_array)
#        styled_image = styled_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
#        styled_image = styled_image.filter(ImageFilter.GaussianBlur(radius=2))
        styled_image = ImageOps.posterize(styled_image, 4)
        styled_image.save(output_path)

    elif style=="test2":
        w, h = image.size
        
        # Parameters for the style (these should come from apply_style args or have defaults)
        num_brush_passes = num_brushes if num_brushes is not None else 3 # e.g., 3 passes
        initial_brush_diameter = brush_size if brush_size is not None else max(5, min(w,h) // 40) # Larger initial brush
        brush_diameter_step = brush_step if brush_step is not None else max(1, initial_brush_diameter // 3) # How much brush shrinks
        base_blur_radius = blur_radius if blur_radius is not None else 3 # Blur for color sampling
        stroke_fervor = fervor if fervor is not None else 1 # Multiplier for number of strokes

        # Create the canvas to paint on. Start with original image or a flat color?
        # For painterly effect, often good to paint onto a copy of the slightly blurred original,
        # or build up from a blank/toned canvas.
        output_image_array = np.array(image.convert("RGB")) # Start with a copy of the original as a base

        # Prepare a source image for color picking - this might be blurred/enhanced differently
        # from the 'image' variable that gets progressively modified if that's the intent.
        # Or, if 'image' is always the color source:
        
        current_pil_image = image.convert("RGB") # Start with the original PIL image for the first pass enhancement

        for pass_idx in range(num_brush_passes):
            current_brush_diameter = max(1, initial_brush_diameter - pass_idx * brush_diameter_step)
            if current_brush_diameter < 1: continue

            # Enhance and blur the PIL image for color sampling in this pass
            enhancer = ImageEnhance.Contrast(current_pil_image)
            enhanced_pil_image = enhancer.enhance(1.2) 
            blurred_pil_for_sampling = enhanced_pil_image.filter(ImageFilter.GaussianBlur(radius=base_blur_radius))
            
            # Determine number of strokes for this pass
            # More strokes for smaller brushes can add detail, or keep density similar
            num_strokes = (w * h // (current_brush_diameter**2)) * stroke_fervor
            # Ensure num_strokes is an int if used in range()
            num_strokes = max(10, int(num_strokes)) # Ensure a minimum number of strokes

            print(f"Pass {pass_idx+1}/{num_brush_passes}: Brush Size={current_brush_diameter}, Strokes={num_strokes}")

            for _ in range(num_strokes):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
        
                # Get color from the specifically prepared blurred image for this pass
                r, g, b = blurred_pil_for_sampling.getpixel((x, y))
                
                # Define brush application area using current_brush_diameter
                brush_radius_int = current_brush_diameter // 2
                for i_col in range(max(0, x - brush_radius_int), min(w, x + brush_radius_int + 1)):
                    for j_row in range(max(0, y - brush_radius_int), min(h, y + brush_radius_int + 1)):
                        dx = i_col - x
                        dy = j_row - y
                        if dx**2 + dy**2 <= brush_radius_int**2: # Use current_brush_diameter
                            new_r_float = r + random.uniform(-20, 20)
                            new_g_float = g + random.uniform(-20, 20)
                            new_b_float = b + random.uniform(-20, 20)
    
                            clamped_r = max(0, min(255, int(new_r_float)))
                            clamped_g = max(0, min(255, int(new_g_float)))
                            clamped_b = max(0, min(255, int(new_b_float)))
                            
                            # Apply to the output_image_array
                            output_image_array[j_row, i_col] = (clamped_r, clamped_g, clamped_b)
            
            # Optional: update current_pil_image for the next pass if you want progressive application
            # current_pil_image = Image.fromarray(output_image_array) # If next pass samples from current result

        styled_image = Image.fromarray(output_image_array)
        # Apply final post-processing filters
        styled_image = styled_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        # Consider if this final blur is always wanted, or if its radius should be different.
        # `actual_blur_radius` here is the one from the very first brush pass's setup in your original code.
        styled_image = styled_image.filter(ImageFilter.GaussianBlur(radius=max(1, base_blur_radius // 2))) # e.g., a smaller finishing blur
        styled_image.save(output_path)

    elif style == "test-orig":
        w, h = image.size
        image_array = np.array(image)
        
        # temp hardcode:
        actual_num_brushes = num_brushes if num_brushes is not None else 2
        actual_num_brushes = min(actual_num_brushes, 3) # clamp to three brushes for now
        actual_brush_size = brush_size if brush_size is not None else 3
        brush_step = 4 
        intensity = 1

        enhancer = ImageEnhance.Contrast(image)
        for idx in range(0, actual_num_brushes):
            bs = actual_brush_size - idx * brush_step
            actual_blur_radius = blur_radius if blur_radius is not None else 4
            intensity = intensity if intensity is not None else 1
            image = enhancer.enhance(1.2)  # Increase contrast by 50%
            image = image.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
            
            for _ in range(w * h // (actual_brush_size**2) * int(intensity)):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
            
                # Get the color of the pixel
                r, g, b = image.getpixel((x, y))
                # Apply brush strokes around the pixel
                for i in range(max(0, x - actual_brush_size // 2), min(w, x + actual_brush_size // 2 + 1)):
                    for j in range(max(0, y - actual_brush_size // 2), min(h, y + actual_brush_size // 2 + 1)):
                        dx = i - x
                        dy = j - y
                        if dx**2 + dy**2 <= (actual_brush_size // 2)**2:
                            # Calculate new color values with random variation
                            new_r_float = r + random.uniform(-20, 20)
                            new_g_float = g + random.uniform(-20, 20)
                            new_b_float = b + random.uniform(-20, 20)
            
                            # Clamp the values to the valid uint8 range [0, 255]
                            clamped_r = max(0, min(255, int(new_r_float)))
                            clamped_g = max(0, min(255, int(new_g_float)))
                            clamped_b = max(0, min(255, int(new_b_float)))
         
                            # Assign the clamped color tuple
                            # Assuming you've made the fix image_array[j, i]
                            image_array[j, i] = (clamped_r, clamped_g, clamped_b)        
        styled_image = Image.fromarray(image_array)
        styled_image = styled_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        styled_image = styled_image.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
        styled_image.save(output_path)    
    else:
        raise ValueError(f"Unknown style: {style}. Supported styles: blur, pixelate, mosaic, impressionist, test, test2, smooth, smooth_more.")

if __name__ == '__main__':
    import traceback
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
        print("Applying test-filter style (default blur)...")
        apply_style("dummy_input.png", "test_outputs/styled_testfilter.png", "test", focus=1.5, num_brushes=2, brush_size=16, brush_step=8, blur_radius=3)
        
        print(f"Example styled images saved in '{os.path.abspath('test_outputs')}' directory.")
        print(f"Dummy input 'dummy_input.png' created in current directory: {os.path.abspath('.')}")

    except ImportError:
        print("Pillow (PIL) library is not installed. This example requires Pillow.")
    except Exception as e:
        print(f"An error occurred during the example: {e}")
        traceback.print_tb(e.__traceback__)  
