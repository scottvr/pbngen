import time
import os

try:
    from pbn.segment import find_stable_label_pixel
    from pbn.segment import xp as active_xp_in_segment # Get the xp used by the imported function
    from pbn.segment import GPU_ENABLED as segment_module_uses_gpu
except ImportError as e:
    print(f"Error importing from pbn.segment: {e}")
    exit()

# ---- Configuration ----
N_ITERATIONS = 100 
HEIGHT = 1000       
WIDTH = 1000        
# DENSITY = 0.5     # control density with xp.random.rand, not used with randint(0,2)

def run_benchmark():
    if segment_module_uses_gpu:
        backend_name = f"CuPy (GPU_ENABLED={segment_module_uses_gpu})"
        # Ensure we are using the CuPy instance from the segment module
        xp = active_xp_in_segment
    else:
        backend_name = f"NumPy (GPU_ENABLED={segment_module_uses_gpu})"
        # Ensure we are using the NumPy instance from the segment module
        xp = active_xp_in_segment

    print(f"--- Starting Benchmark for find_stable_label_pixel ---")
    print(f"Using backend: {backend_name}")
    print(f"Number of iterations: {N_ITERATIONS}")
    print(f"Array dimensions: {HEIGHT}x{WIDTH} ({HEIGHT*WIDTH} elements)")
    print("Generating random binary masks for each iteration...")

    durations = []
    
    for i in range(N_ITERATIONS):
        # Generate random binary mask (0s and 1s)
        # This creates an array with roughly 50% density of 1s.
        # Data is generated using the 'xp' instance (CuPy or NumPy)
        # that find_stable_label_pixel will use.
        dummy_mask = xp.random.randint(0, 2, size=(HEIGHT, WIDTH), dtype=xp.uint8)

        # Synchronize GPU before starting timer if using CuPy for accurate timing
        if segment_module_uses_gpu and hasattr(xp, 'cuda') and hasattr(xp.cuda, 'Stream'):
            xp.cuda.Stream.null.synchronize()
        
        start_time = time.perf_counter()

        # Call the function
        # The imported_find_stable_label_pixel will use the 'xp' from its own module's scope
        result_x, result_y = find_stable_label_pixel(dummy_mask)

        # Synchronize GPU after operation if using CuPy
        if segment_module_uses_gpu and hasattr(xp, 'cuda') and hasattr(xp.cuda, 'Stream'):
            xp.cuda.Stream.null.synchronize()
            
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        durations.append(duration)
        
        # Print progress, e.g., every 10% or every iteration if N_ITERATIONS is small
        print_interval = max(1, N_ITERATIONS // 10)
        if (i + 1) % print_interval == 0 or N_ITERATIONS <= 10 :
            print(f"  Iteration {i+1}/{N_ITERATIONS} done. Result: ({result_x}, {result_y}). Time: {duration:.6f}s")

    # ---- Results ----
    if not durations:
        print("No iterations were run.")
        return

    total_time = sum(durations)
    avg_time = total_time / N_ITERATIONS
    min_time = min(durations)
    max_time = max(durations)
    
    # Calculate iterations per second (throughput)
    throughput = N_ITERATIONS / total_time if total_time > 0 else float('inf')

    print("\n--- Benchmark Results Summary ---")
    print(f"Backend used: {backend_name}")
    print(f"Total iterations: {N_ITERATIONS}")
    print(f"Array dimensions: {HEIGHT}x{WIDTH}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per call: {avg_time:.6f} seconds ({avg_time*1000:.3f} ms)")
    print(f"Min time per call: {min_time:.6f} seconds ({min_time*1000:.3f} ms)")
    print(f"Max time per call: {max_time:.6f} seconds ({max_time*1000:.3f} ms)")
    print(f"Throughput: {throughput:.2f} calls/second")

    # with open("benchmark_results.txt", "a") as f:
    #     f.write(f"Backend: {backend_name}, Avg: {avg_time:.6f}s, Total: {total_time:.4f}s, N: {N_ITERATIONS}\n")

if __name__ == "__main__":
    run_benchmark()