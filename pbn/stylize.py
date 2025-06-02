from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageOps
import numpy as np
import random
import os 
import typer # for typer.echo
import math
from typing import Optional # For type hinting

def apply_filter(
    input_path: str, 
    filter_name: str,  # Changed from 'filter' to 'filter_name' to avoid conflict with built-in
    blur_radius: Optional[int] = None, 
    pixelate_block_size: Optional[int] = None, 
    mosaic_block_size: Optional[int] = None, 
    brush_step: Optional[int] = None, 
    brush_size: Optional[int] = None, 
    edge_strength: Optional[float] = None, # Added from pbngen call
    num_brushes: Optional[int] = None, 
    fervor: Optional[int] = None, # Added from pbngen call
    focus: Optional[int] = None  # Added from pbngen call
) -> Optional[Image.Image]:
    """
    Applies a specified visual filter to an input image and returns the filtered PIL Image.

    Args:
        input_path (str): Path to the input image file.
        filter_name (str): The name of the filter to apply.
        ... (other filter-specific args)

    Returns:
        Optional[PIL.Image.Image]: The filtered image as a PIL Image object, or None on error.

    Raises:
        ValueError: If the input file is not found, cannot be opened, or if an unknown filter is specified.
    """
    try:
        image = Image.open(input_path)
    except FileNotFoundError:
        raise ValueError(f"Error: Input file not found at {input_path}")
    except Exception as e:
        raise ValueError(f"Error opening image {input_path}: {e}")

    filtered_image: Optional[Image.Image] = None # Initialize

    if filter_name == "blur":
        actual_blur_radius = blur_radius if blur_radius is not None else 4
        filtered_image = image.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
    
    elif filter_name == "smooth":
        filtered_image = image.filter(ImageFilter.SMOOTH)
        
    elif filter_name == "smooth_more":
        filtered_image = image.filter(ImageFilter.SMOOTH_MORE)

    elif filter_name == "pixelate":
        actual_block_size = pixelate_block_size if pixelate_block_size is not None else 10
        actual_block_size = max(1, actual_block_size) # Ensure block_size is at least 1
        if image.width < actual_block_size or image.height < actual_block_size:
            typer.echo(f"Warning: Pixelate block size ({actual_block_size}) is larger than image dimensions. Using smaller block size.")
            actual_block_size = max(1, min(image.width, image.height, actual_block_size))

        img_small = image.resize((image.width // actual_block_size, image.height // actual_block_size), Image.Resampling.BILINEAR)
        filtered_image = img_small.resize(image.size, Image.Resampling.NEAREST)
        # typer.echo(f"Stylize: 'pixelate' filter with block_size {actual_block_size}.") # Debug

    elif filter_name == "mosaic":
        w, h = image.size
        actual_mosaic_block_size = mosaic_block_size if mosaic_block_size is not None else max(4, min(w, h) // 64) # Default based on image size
        actual_mosaic_block_size = max(1, actual_mosaic_block_size) # Ensure positive
        if w < actual_mosaic_block_size or h < actual_mosaic_block_size:
            typer.echo(f"Warning: Mosaic block size ({actual_mosaic_block_size}) is larger than image dimensions. Adjusting.")
            actual_mosaic_block_size = max(1, min(w, h, actual_mosaic_block_size))
            
        # typer.echo(f"Stylize: 'mosaic' filter with block_size {actual_mosaic_block_size}.") # Debug
        
        # Resize down with NEAREST to get distinct blocks
        img_small = image.resize((w // actual_mosaic_block_size, h // actual_mosaic_block_size), resample=Image.Resampling.NEAREST)
        # Resize back up with BICUBIC to blend the blocks smoothly
        filtered_image = img_small.resize((w, h), resample=Image.Resampling.BICUBIC)


    elif filter_name == "impressionist":
        w, h = image.size
        image_rgb = image.convert("RGB")
        actual_blur_radius = blur_radius if blur_radius is not None else 10
        
        blurred_source = image_rgb.filter(ImageFilter.GaussianBlur(radius=actual_blur_radius))
        output_canvas = Image.new("RGB", (w, h), (255, 255, 255)) 
        draw = ImageDraw.Draw(output_canvas)

        # Dynamic brush sizes based on image dimension
        base_dim = min(w,h)
        brush_diameters_to_use = [
            max(3, int(base_dim * 0.03)), 
            max(2, int(base_dim * 0.015)), 
            max(1, int(base_dim * 0.0075))
        ]
        # Fallback if image is very small
        if not any(d > 0 for d in brush_diameters_to_use): brush_diameters_to_use = [3,2,1]


        for brush_d in brush_diameters_to_use:
            if brush_d < 1: continue
            step = max(1, brush_d // 2) # 50% overlap for strokes

            paint_locations = []
            for y_base in range(0, h, step):
                for x_base in range(0, w, step):
                    jitter_amount = step // 3 # Reduced jitter relative to step
                    jitter_x = random.randint(-jitter_amount, jitter_amount)
                    jitter_y = random.randint(-jitter_amount, jitter_amount)
                    center_x = max(0, min(w - 1, x_base + jitter_x))
                    center_y = max(0, min(h - 1, y_base + jitter_y))
                    paint_locations.append((center_x, center_y))
            
            random.shuffle(paint_locations)

            for px, py in paint_locations:
                try:
                    color = blurred_source.getpixel((px, py))
                except IndexError: color = (0,0,0) 
                
                half_brush_d = brush_d / 2.0
                x0, y0 = px - half_brush_d, py - half_brush_d
                x1, y1 = px + half_brush_d, py + half_brush_d
                draw.ellipse([(x0, y0), (x1, y1)], fill=color)
                
        filtered_image = output_canvas

        
    elif filter_name.startswith("painterly-"):
        level = filter_name.split("-")[-1]
        w, h = image.size
        
        # Convert image to RGB and then to NumPy array
        # Ensure that we operate on an RGB version for consistent color handling
        image_rgb = image.convert("RGB")
        output_image_array = np.array(image_rgb) # Base for modifications

        # Parameters based on level (lo, med, hi)
        if level == "lo":
            actual_num_brushes = num_brushes if num_brushes is not None else 1
            initial_brush_diameter = brush_size if brush_size is not None else max(8, min(w, h) // 30)
            brush_diameter_step = brush_step if brush_step is not None else max(1, initial_brush_diameter // 3)
            base_blur_radius = blur_radius if blur_radius is not None else 5
            stroke_fervor = fervor if fervor is not None else 1 # Lower stroke density
            contrast_enhancement = 1.1
            final_posterize_bits = 5 # More posterization for 'lo'
            final_sharpen = False
            final_blur_factor = 0.6
        elif level == "med":
            actual_num_brushes = num_brushes if num_brushes is not None else 2
            initial_brush_diameter = brush_size if brush_size is not None else max(6, min(w, h) // 40)
            brush_diameter_step = brush_step if brush_step is not None else max(1, initial_brush_diameter // 3)
            base_blur_radius = blur_radius if blur_radius is not None else 4
            stroke_fervor = fervor if fervor is not None else 2 # Medium stroke density
            contrast_enhancement = 1.15
            final_posterize_bits = 6
            final_sharpen = True
            final_blur_factor = 0.5
        elif level == "hi": # Corresponds to painterly-hi
            actual_num_brushes = num_brushes if num_brushes is not None else 3
            initial_brush_diameter = brush_size if brush_size is not None else max(5, min(w,h) // 50)
            brush_diameter_step = brush_step if brush_step is not None else max(1, initial_brush_diameter // 4)
            base_blur_radius = blur_radius if blur_radius is not None else 3
            stroke_fervor = fervor if fervor is not None else 3 # Higher stroke density
            contrast_enhancement = 1.2
            final_posterize_bits = 7 # Less posterization for 'hi'
            final_sharpen = True
            final_blur_factor = 0.4

        else:
            raise ValueError(f"Unknown painterly level: {level}. Supported: lo, med, hi.")

        current_pil_image_for_sampling = image_rgb # Start with the initial RGB image for sampling

        for pass_idx in range(actual_num_brushes):
            current_brush_diameter = max(1, initial_brush_diameter - pass_idx * brush_diameter_step)
            if current_brush_diameter < 1: continue

            # Enhance and blur the *current* PIL image for color sampling in this pass
            enhancer = ImageEnhance.Contrast(current_pil_image_for_sampling)
            enhanced_pil_image = enhancer.enhance(contrast_enhancement) 
            blurred_pil_for_sampling = enhanced_pil_image.filter(ImageFilter.GaussianBlur(radius=max(1, int(base_blur_radius * (1 - pass_idx * 0.1))))) # Slightly reduce blur for later passes
            
            num_strokes = (w * h // (current_brush_diameter**2)) * stroke_fervor
            num_strokes = max(10, int(num_strokes))

            typer.echo(f"  Painterly Pass {pass_idx+1}/{actual_num_brushes} ({level}): Brush Size={current_brush_diameter}, Strokes={num_strokes}")

            for _ in range(num_strokes):
                # Pick random center for stroke
                x_center = random.randint(0, w - 1)
                y_center = random.randint(0, h - 1)
        
                # Get color from the (potentially specifically prepared) blurred image for this pass
                r, g, b = blurred_pil_for_sampling.getpixel((x_center, y_center))
                
                # Apply circular brush stroke to output_image_array
                brush_radius_int = current_brush_diameter // 2
                # Define the bounding box for the stroke to iterate over
                x_start = max(0, x_center - brush_radius_int)
                x_end = min(w, x_center + brush_radius_int + 1)
                y_start = max(0, y_center - brush_radius_int)
                y_end = min(h, y_center + brush_radius_int + 1)

                for i_col in range(x_start, x_end):
                    for j_row in range(y_start, y_end):
                        # Check if the pixel (j_row, i_col) is within the circle
                        dx = i_col - x_center
                        dy = j_row - y_center
                        if dx*dx + dy*dy <= brush_radius_int*brush_radius_int:
                            # Add some random variation to the color
                            new_r_float = r + random.uniform(-15, 15) # Reduced variation for finer control
                            new_g_float = g + random.uniform(-15, 15)
                            new_b_float = b + random.uniform(-15, 15)
    
                            clamped_r = max(0, min(255, int(new_r_float)))
                            clamped_g = max(0, min(255, int(new_g_float)))
                            clamped_b = max(0, min(255, int(new_b_float)))
                            
                            output_image_array[j_row, i_col] = (clamped_r, clamped_g, clamped_b)
            
            # Update the image used for sampling in the next pass to be the result of the current pass
            # This makes the effect build upon itself.
            current_pil_image_for_sampling = Image.fromarray(output_image_array, 'RGB')

        filtered_image_pil = Image.fromarray(output_image_array, 'RGB')
        
        # Post-processing
        if final_posterize_bits > 0 and final_posterize_bits < 8:
            filtered_image_pil = ImageOps.posterize(filtered_image_pil, final_posterize_bits)
        if final_sharpen:
            filtered_image_pil = filtered_image_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        if base_blur_radius > 0 and final_blur_factor > 0:
             final_blur_radius = max(1, int(base_blur_radius * final_blur_factor))
             filtered_image_pil = filtered_image_pil.filter(ImageFilter.GaussianBlur(radius=final_blur_radius))
        
        filtered_image = filtered_image_pil

    else:
        raise ValueError(f"Unknown filter: {filter_name}. Supported: blur, pixelate, mosaic, impressionist, painterly-[lo,med,hi], smooth, smooth_more.")

    # No longer saves the image here, returns the PIL Image object
    return filtered_image


if __name__ == '__main__':
    import traceback
    print("Running stylize.py example...")
    if not os.path.exists("test_outputs"):
        os.makedirs("test_outputs")
    
    try:
        img = Image.new("RGB", (256, 256), color = (150, 120, 200))
        draw_test = ImageDraw.Draw(img)
        draw_test.rectangle([(50,50), (150,150)], fill=(200,50,50))
        draw_test.ellipse([(100,100), (200,200)], fill=(50,200,50))
        img.save("dummy_input.png")

        filters_to_test = {
            "blur": {"blur_radius": 5},
            "impressionist": {"blur_radius": 8},
            "pixelate": {"pixelate_block_size": 16},
            "mosaic": {"mosaic_block_size": 20},
            "painterly-lo": {"num_brushes": 1, "brush_size": 20, "blur_radius": 3},
            "painterly-med": {"num_brushes": 2, "brush_size": 15, "blur_radius": 2},
            "painterly-hi": {"num_brushes": 3, "brush_size": 10, "blur_radius": 1, "fervor": 2},
            "smooth": {},
            "smooth_more": {}
        }

        for filt_name, filt_params in filters_to_test.items():
            print(f"Applying '{filt_name}' filter...")
            output_filename = f"test_outputs/filtered_{filt_name.replace('-', '_')}.png"
            try:
                # Call apply_filter which now returns a PIL image
                result_img = apply_filter("dummy_input.png", filt_name, **filt_params)
                if result_img:
                    result_img.save(output_filename) # Save the returned image
                    print(f"  Saved to {output_filename}")
                else:
                    print(f"  Filter '{filt_name}' did not return an image.")
            except ValueError as ve:
                print(f"  Error applying filter '{filt_name}': {ve}")
            except Exception as e_filt:
                print(f"  Unexpected error with filter '{filt_name}': {e_filt}")
                traceback.print_tb(e_filt.__traceback__)
        
        print(f"\nExample filtered images saved in '{os.path.abspath('test_outputs')}' directory.")
        print(f"Dummy input 'dummy_input.png' created in current directory: {os.path.abspath('.')}")

    except ImportError:
        print("Pillow (PIL) library is not installed. This example requires Pillow.")
    except Exception as e:
        print(f"An error occurred during the example: {e}")
        traceback.print_tb(e.__traceback__)