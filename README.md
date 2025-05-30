# PbNgen: the Python Paint-By-Number Generator

The **LeadEngine** (aka [Pb](https://wikipedia.com/wiki/lead)Ngen, that's "lead" as in paint) is a command-line tool that converts any image into a printable, paint-by-number guide. It reduces images to a fixed color palette, segments the result into paintable regions, overlays numeric labels, and generates both raster and vector outputs along with a color swatch legend.

## Features

- Converts any image into paint-by-number format.
- Two-stage color quantization:
    - Optional initial global color reduction using Bits Per Pixel (`--bpp`).
    - Final PBN palette generation using a specific number of colors (`--num-colors`).
- Style preprocessing: `blur`, `pixelate`, `mosaic`, and a poorly-named `"impressionist"` filter.
- Complexity presets for beginner to master-level detail.
- Label placement strategies: `diagonal`, `centroid`, `stable`.
- Custom font support for overlays and legend.
- Vector (SVG) and raster (PNG) output.
- Optional fixed-palette matching from a source image (can also be pre-quantized with `--bpp`).
- Fully command-line driven with rich configuration.
- GPU (Nvidia CUDA) support!

## Installation

Requires Python 3.7+

```bash
pip install -r requirements.txt
```

**Note: if using GPU, please ensure you have CUDA correctly configured, and you will need to `pip install cupy` if it isn't already in your environment.**

Windows users: see [section on GPU under WSL](#note-concerning-scikit-learn-cuml-gpu-acceleration-on-windows)
## Usage

The script is invoked directly, followed by options, the input image file, and the output directory.

**General Syntax:**
```bash
python pbngen.py [OPTIONS] INPUT_FILE OUTPUT_DIRECTORY
```

### Quick Start

```bash
python pbngen.py my_photo.jpg ./pbn_output
```
This command processes `my_photo.jpg`, saves outputs to the `./pbn_output` directory, and uses default settings (typically 12 colors for the PBN palette, intermediate complexity).

[A Basic Example complete with input and output images is available on the Wiki](https://github.com/scottvr/pbngen/wiki/Basic-Example) 

See [Tips for Best Results](#tips-for-best-results) further down in this README.

### More Complex Example

```bash
python pbngen.py "Vacation Pic.png" "./Vacation PBN" \
  --style mosaic \
  --bpp 8 \
  --num-colors 16 \
  --complexity master \
  --palette-from "artist_palette.png" \
  --font-path "./fonts/Arial.ttf" \
  --label_strategy stable \
  --blobbify \
  --blob-min 5 \
  --dpi 300 \
  --yes
```
This example:
- Processes `"Vacation Pic.png"` and outputs to `"./Vacation PBN"`.
- Applies a `mosaic` style.
- First, pre-quantizes the input image (and `artist_palette.png`) to an 8-bit (256 colors) palette using `--bpp 8`.
- Then, generates the final PBN using 16 colors (`--num-colors 16`), derived from the pre-quantized `artist_palette.png`.
- Uses the `master` complexity settings (which might influence defaults if `--num-colors` wasn't specified).
- Uses a custom font (`Arial.ttf`) and the `stable` label mode.
- Applies `blobbify` for a painterly effect with 5mm² minimum blob size, using 300 DPI for calculations.
- Overwrites existing files without prompting.

## Output Files

Assuming `OUTPUT_DIRECTORY` is `out/`:

- **Final PBN Files:**
    - `out/pbn_guide-ncolor_quantized.png`: The input image reduced to the final PBN color palette (e.g., 12 colors). This is the image used for segmentation and is a fair representation of what your painting should look like when completed.
    - `out/raster-pbn_canvas.png`: The main raster (PNG) paint-by-number guide with outlines and numeric labels.
    - `out/vector-pbn_canvas.svg`: A vector (SVG) version of the above (if not `--raster-only`). This is what you'll want to use if you are sending the image to a print shop for otuput to a large canvas since vector graphics scale without losing fidelity.
    - `out/palette-pbn_legend.png`: A color swatch legend mapping numbers to the final PBN palette colors (if not `--skip-legend`).

- **Intermediate Files (may be present in the output directory):**
    - `out/styled_input.png`: If a `--style` is applied.
    - `out/input-bpp_quantized.png`: If `--bpp` is used, this is the input image after the initial BPP color reduction.
    - `out/bpp_quantized_palette_input.png`: If `--bpp` and `--palette-from` are used, this is the palette source image after BPP reduction.

 **Note: the intermediate files can be automatically cleaned up (deleted) by passing the --no-cruft flag on the command-line.**

## Tips for Best Results

- for complex images, `--style blur` almost always helps a ton.
- for simple, clean low-color images (like logos) don't blur first
- experiment; try both and see which gives better results
- **likewise for the rest of the options below. Try a smaller font size. Try mixing and matching complexity presets with non-default font-size, tile-spacing, etc.**
- Please read the documentation to the end and try a few settings before opening an Issue
- If it really seems that pbngen can't make a good pbn canvas image for you, please *DO* open an Issue; attach the image you are trying to process along with the command-lines you've tried.

-----

## Options

| Flag                          | Description                                                                                                                               | Default        |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| `INPUT_FILE`                  | (Positional) Path to the input image file.                                                                                                | **Required** |
| `OUTPUT_DIRECTORY`            | (Positional) Path to the directory where output files will be saved.                                                                      | **Required** |
| `--complexity TEXT`           | Preset detail level: `beginner`, `intermediate`, `master`. Affects defaults for `num-colors`, `tile-spacing`, `font-size`.                | `None`         |
| `--style TEXT`                | Preprocessing style to apply to the input image: `blur`, `pixelate`, `mosaic`, `impressionist`, `test`, `test2`.                                                            | `None`         |
| `--min-region-area INTEGER` | Minimum pixel area for a color region to be processed and included in the final output. Regions smaller than this will be discarded. Useful for controlling detail and noise. Must be a positive integer. | 50 (from segmentation logic) |
| `--small-region-strategy` | This allows the selection of which labeling strategy (`centroid`, `stable`, `diagonal`) should be tried when attempting to place a label in a region that is shorter or narrower than the `tile-spacing` setting. | stable |
| `--num-colors INTEGER`        | Final number of colors for the PBN palette.                                                                                               | 12             |
| `--bpp INTEGER`               | Bits Per Pixel (1-8) for an *initial* color depth reduction (e.g., 8 for 256 colors). Applied before `--num-colors` processing.            | `None`         |
| `--palette-from FILE_PATH`    | Path to an image to extract a fixed palette from for the final PBN colors. This image is also subject to `--bpp` pre-quantization.        | `None`         |
| `--font-path FILE_PATH`       | Path to a `.ttf` font file for labels and legend.                                                                                         | System default |
| `--font-size INTEGER`         | Base font size for overlay labels and legend text.                                                                                        | 12             |
| `--label_strategy TEXT`           | Label placement strategy: `diagonal`, `centroid`, `stable`.                                                                               | `diagonal`     |
| `--tile-spacing INTEGER`      | Approximate distance (pixels) between repeated number labels in a large region.                                                           | 30             |
| `--swatch-size INTEGER`       | Width/height (pixels) of each color swatch in the legend.                                                                                 | 40             |
| `--skip-legend` | If set, skips generating the palette legend image.                                                                               | `False`        |
| `--raster-only` | Only generate raster output (labeled PNG); skip vector SVG output.                                                                 | `False`        |
| `--yes` / `-y`                | Automatically overwrite existing output files without prompting.                                                                          | `False`        |
| `--blobbify` | Enable painterly splitting of color regions into smaller, more organic 'blobs'.                                                          | `False`        |
| `--blob-min INTEGER`          | Minimum blob area in mm² (used if `--blobbify` is active).                                                                                | 3              |
| `--blob-max INTEGER`          | Maximum blob area in mm² (used if `--blobbify` is active).                                                                                | 30             |
| `--min-label-font INTEGER`    | Minimum font size allowed for blob labeling if `--blobbify` is active.                                                                    | 8              |
| `--no-interpolate-contours` |  Do not interpolate contour lines for smoother appearance. Disable for simpler SVGs.                                    | `True`         |
| `--dpi INTEGER`               | Dots Per Inch (DPI) for mm² to pixel conversion (used with `--blobbify`). Overrides DPI from image metadata if provided.                | Auto or 96     |
| `-h`, `--help`                | Show the help message and exit.                                                                                                           |                |



### Complexity Presets

| Name         | PBN Colors | Tile Spacing | Font Size |
|--------------|------------|--------------|-----------|
| `beginner`   | 6          | 40px         | 14        |
| `intermediate`| 12         | 30px         | 12        |
| `master`     | 24         | 20px         | 10        |
*(These presets set defaults; explicit options like `--num-colors` will override them.)*

### Style Options

| Name       | Description                              |
|------------|------------------------------------------|
| `blur`     | Applies a Gaussian blur.                 |
| `pixelate` | Chunky low-resolution pixelation.        |
| `mosaic`   | Pixelate and upscale for blended effect. |
| `impressionist` | Make the source image look more "painterly" before beginning the PBN conversion. |
| `test`  | A slightly-better attempt at the above. Introduces the --brush-* options elsewhere described. |
| `test2` | About on par with the previous, but responds to `fervor` rather than `focus` and behaves differently in the lack of --brush-* arguments |


### Styling Options and Parameters

The `--style TEXT` option allows you to apply a visual preprocessing effect to your input image. The available styles are `blur`, `pixelate`, and `mosaic`. You can further control the intensity or characteristics of these styles using the following parameters:

| Style Selected      | Parameter Flag              | Description                                                                                                        | Default Behavior (if flag not used)                                    |
|---------------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `--style blur`      | `--blur-radius INTEGER`     | Sets the radius for the Gaussian blur effect. Higher values increase blurriness. Must be a positive integer.        | `4` (as defined in the `stylize` module)                             |
| `--style pixelate`  | `--pixelate-block-size INTEGER` | Sets the size (in pixels) of the blocks for the pixelation effect. Must be a positive integer.                  | Dynamically calculated: `max(4, min(image_width, image_height) // 64)` |
| `--style mosaic`    | `--mosaic-block-size INTEGER` | Sets the size (in pixels) of the blocks for the mosaic effect. Must be a positive integer.                         | Dynamically calculated: `max(4, min(image_width, image_height) // 64)` |
| `--num-brushes INTEGER`       | Number of brushes to use (in order of decreasing size) in the painterly style filters that make multiple passes                                       | 1 - 3    |
| `--brush-size INTEGER`        | Size (in px) of the first (largest) brush used in a style filter |    Default is calculated based on image size. I should make this integer value an area in mm^2 really.                 |
| `--brush-step INTEGER`        | how many units (px) to step down between brush changes, where applicable.                     | Calculated based on `num-brushes` and `brush-size`.   |
| `--focus INTEGER`             | How "tidy" the pre-pbn style filter brush strokes are. Lower is more painterly and as a bonus is faster than a high number.   | 1         |
| `--fervor INTEGER`            | How "manic" the pre-pbning style filter brush strokes are. Higher is more intense and takes longer.   |   1       |

-----

**How to Use Styling Options:**

First, select a style with `--style <style_name>`, then optionally add its corresponding parameter flag. For example:

* To apply a blur with a radius of 7:
    ```bash
    python pbngen.py input.jpg ./output --style blur --blur-radius 7
    ```

* To apply pixelation with a block size of 10 pixels:
    ```bash
    python pbngen.py input.jpg ./output --style pixelate --pixelate-block-size 10
    ```

* To apply the mosaic effect with its default (dynamically calculated) block size:
    ```bash
    python pbngen.py input.jpg ./output --style mosaic
    ```

If you specify a style parameter flag (e.g., `--blur-radius`) without the corresponding `--style` flag (e.g., `--style blur`), the parameter flag will be ignored as no style is being applied. The parameter flags are only active when their associated style is selected.

------

### Specifying Physical Canvas Size for Printing (`--canvas-size`)

To prepare your paint-by-number output for printing on a specific physical canvas size (e.g., an 8x10 inch sheet, A4 paper), you can use the `--canvas-size` option.

| Flag                          | Description                                                                                                                                                              | Example Usage                     |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `--canvas-size TEXT`          | Sets the desired physical dimensions for the final output. The image content will be scaled to fit these dimensions (maintaining aspect ratio) and centered on a white background of this size. The format is `WIDTHxHEIGHTUNIT` (no spaces around `x`). Supported units: `in` (inches), `cm` (centimeters), `mm` (millimeters). This option relies on an accurately set DPI (see `--dpi`). | `--canvas-size "10x8in"` <br> `--canvas-size "29.7x21cm"` <br> `--canvas-size "250x200mm"` |

**How it Works:**

1.  You provide the desired physical dimensions and unit (e.g., `"10x8in"`; units can be `in`, `cm`, or `mm`).
2.  The script uses the **effective DPI** (Dots Per Inch) to convert these physical dimensions into pixel dimensions for the output canvas.
    * The effective DPI is determined by:
        1.  The value you provide with the `--dpi` flag (e.g., `--dpi 300`).
        2.  If `--dpi` is not set, the DPI from the input image's metadata (if available).
        3.  If neither is available, a default of 96 DPI is used.
3.  The original image (after any styling or BPP pre-quantization) is then intelligently resized to fit within these target pixel dimensions while maintaining its original aspect ratio.
4.  A new canvas is created with the exact target pixel dimensions and a white background.
5.  The resized image content is centered onto this new canvas.
6.  All subsequent processing (final PBN color quantization, segmentation, labeling, SVG/PNG output) occurs on this new, dimensionally-correct canvas.

**Example:**

To generate a monochrome paint-by-number that will be printed on a 12x16 inch canvas at 300 DPI:

```bash
python pbngen.py my_image.jpg ./output_12x16 \
  --canvas-size "12x16in" \
  --dpi 300 \
  --num-colors 2
```
-----

### Label Modes (Placement Strategies)

-   `diagonal` (default): Fast, places labels on a diagonal grid within regions. Can sometimes be dense.
-   `centroid`: Places a single label at the geometric center of each region. Best for simple, convex shapes.
-   `stable`: Finds the most "stable" point within each region (deepest inside the color). Often gives good placement but is significantly slower.

-----

### Understanding `--bpp` vs `--num-colors`

These two options control color reduction at different stages:

1.  **`--bpp INTEGER` (Initial Pre-quantization):**
    * If provided, this option performs an *initial* global color reduction on the input image (and the `--palette-from` image, if used).
    * It quantizes the image(s) to `2^BPP` colors (e.g., `--bpp 8` results in 256 colors).
    * This step is useful for:
        * Standardizing the color palette of diverse input images before PBN-specific processing.
        * Working within a constrained color space if the original image has an extremely high number of colors.
        * Ensuring that a user-supplied palette (via `--palette-from`) is considered from within this same reduced color space.
    * The output of this stage (if `--bpp` is used) becomes the input for the next stage.

2.  **`--num-colors INTEGER` (Final PBN Palette Size):**
    * This determines the number of colors that will be in your *final paint-by-number palette*.
    * The PBN quantization process (using scikit-learn's KMeans or a fixed palette from `--palette-from`) will reduce the (potentially BPP-pre-quantized) image to this many colors.
    * This is the number of distinct paints you'll need.

**Example Flow:**
`INPUT_IMAGE (e.g., 24-bit color)`
  `--bpp 8` → `INTERMEDIATE_IMAGE_1 (256 colors)`
    `--num-colors 16` (using KMeans on INTERMEDIATE_IMAGE_1) → `FINAL_PBN_IMAGE (16 colors for painting)`

If `--palette-from MY_PALETTE.PNG` is also used with `--bpp 8` and `--num-colors 16`:
`INPUT_IMAGE` → `INTERMEDIATE_INPUT (256 colors)`
`MY_PALETTE.PNG` → `INTERMEDIATE_PALETTE_SOURCE (256 colors)`
  Then, 16 colors are extracted from `INTERMEDIATE_PALETTE_SOURCE` and used to quantize `INTERMEDIATE_INPUT` to the `FINAL_PBN_IMAGE (16 colors)`.

# How it works (Big Picture)

### Pipeline Diagram

The image processing pipeline involves several stages:

```mermaid
graph TD
    A["Input Image"] -->|Optional: --style| B["Styled Image (if any)"];
    B -->|Optional: --bpp| C["BPP Pre-quantized Image (e.g., 256 colors)"];
    C -->|--num-colors & optional --palette-from| D["Final PBN Quantized Image (e.g., 12 colors)"];
    D --> E["Segment Regions & Collect Primitives"];
    E -->|Optional: --blobbify| F["Blobbified Primitives"];
    F --> G["Render Labeled Raster (PNG)"];
    F --> H["Render Vector Output (SVG)"];
    D --> I["Generate Palette Legend (PNG)"];

    subgraph "Input Handling"
        A
        B
    end
    subgraph "Color Reduction"
        C
        D
    end
    subgraph "Segmentation"
        E
        F
    end
    subgraph "Output Generation"
        G
        H
        I
    end
```

As you can see, the vector representation (of segmented primitives) is the source for the raster image, so eliding SVG output it won't save any noticeable time. 

-----

# Experimental Features

## Painterly Output with `--blobbify`

The `--blobbify` feature allows for a more artistic, "painterly" result by subdividing large, uniform color regions into smaller, more organically shaped "blobs." This can simulate the appearance of individual brushstrokes.

### Description

When `--blobbify` is used, after the initial color segmentation, contiguous regions of the same color are further processed. They are broken down into smaller blobs, with the size and number of blobs influenced by the `--blob-min` and `--blob-max` area settings (in mm², converted to pixels using DPI).

Blobs that are too small to legibly fit a label (based on `--min-label-font`) are merged with suitable adjacent blobs to ensure all painted areas can be identified.

### Blobbify Flags

| Flag                | Description                                                                 | Default |
|---------------------|-----------------------------------------------------------------------------|---------|
| `--blobbify`        | Enable painterly splitting of color regions.                                | `False` |
| `--blob-min`        | Minimum blob area in mm² (converted to pixels using DPI).                   | 3       |
| `--blob-max`        | Maximum blob area in mm² (converted to pixels using DPI).                   | 30      |
| `--min-label-font`  | Minimum font size (points) required for a blob to receive its own label.    | 8       |
| `--dpi`             | DPI used for mm² to pixel conversion for blob sizing. Uses image metadata or defaults to 96 if not set. | Auto/96 |

### Blobbify Example

```bash
python pbngen.py "landscape.png" "./landscape_pbn_blobs" \
  --num-colors 20 \
  --blobbify \
  --blob-min 4 \
  --blob-max 25 \
  --min-label-font 9 \
  --dpi 150 \
  --font-path "./fonts/ComicSans.ttf" \
  --yes
```
This generates a PBN for `landscape.png` with 20 final colors, where regions are blobbified into sizes roughly between 4mm² and 25mm² (calculated at 150 DPI), and labels use Comic Sans with a minimum size of 9pt.

-----

## Hardware Support (GPU Acceleration)

Some operations are computationally-expensive enough that for a large and complex source image, no amount of CPU is going to be performant enough to complete in a "while-you-wait" timeframe. In the interest of more-immediate gratification, repetitive calculations have been parallelized with support for running on NVIDIA CUDA-capable GPUs. 

If installed, PbNgen will automagically use CuPy in place of NumPy and SciPy for many operations, provided it can detect a working CUDA environment. It has been tested with CUDA 12.9.

As already mentioned, you need to have a working CUDA installation on which to run PbNgen with acceleration, along with cupy in the environment where pbngen will run.  If you want to run blobbify on an image of any sort of large input or output canvas size, or if your image naturally has many small regions of colors, you may find that CPU-only version is too painfully slowto bear.

This link is to instructions that will help you get it set up if you need it: [Official CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html)

PbNgen will attempt to use the GPU if all prerequisites are met and CuPy is functioning, otherwise it will fallback to CPU.  A message will be printed to stdout on startup whether it will use CPU or GPU.

Further, some of SciKit Learn's (sk-learn) functionality can be accelerated by initializing [NVIDIA RAPIDS CuML](https://docs.rapids.ai/install/#wsl2) as per the docs linked here. 

To get scikit-learn to use these accelerated functions you must wrap your pbngen invocation with cuml like so: 

``` bash 
python -m cuml.accel pbngen.py input_file output_dir --your --args -here ...
``` 

### Running on Google Colab

``` jupyter
% load_ext cuml.accel
% !python pbngen.py inputimage.png ./output_dir
```

There is a nice and easy [notebook here](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/rapids-pip-colab-template.ipynb) that will get you up and running quickly without all of the hassle that goes along with getting a working nvidia dev environment (which is especially sticky on Windows.) Unfortunately, pbngen doesn't yet gain a while lot just yet from this additional scikit acceleration, but after seeing a 50x improvement in the stable placement algorithm after parallelizing and making the code use cupy if available, I am eager to see how much more there may be to gain with cuml.

**Note Concerning CuML GPU Acceleration on Windows**

Because NVIDIA RAPIDS is compiled for a linux target, you'll have to run it under WSL2 to get the performance increase CuML offers if you are on a Windows system. The aforementioned (and linked) RAPIDS Colab might be what I'd suggest since it's already Linux, the setup is already in the notebook, Google's GPUs are likely bigger than yours, and it took less than a minute (the notebook warns it should take ~5 min, but it wasn't even 60 seconds when I tried it.)

In short you will need the NVIDIA **Windows** driver, WSL2 working on your Windows machine, and basic Linux knowledge for the distro you installed in WSL2.
See these links for details:

- [CUDA in WLS2 User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Installing RAPIDS under WSL2](https://docs.rapids.ai/install/#wsl2)

-----

## Credits

PbNgen is the work of [scottvr](https://github.com/scottvr) and is a rewrite of some old bash, perl and ImageMagick voodoo I used to use to accomplish mostly the same, albeit without as good a resultant product. PbNgen is still undergoing active development so if you have a feature request or bug report, please let me know via Issues. If you have an ehancement, please send me a PR on Github.

The stable label ranking algorithm was shamelessly ripped from [PBNify](https://pbnify.com), a browser app that accomplishes roughly the same thing as PbNgen albeit in a different interface and without hardware acceleration or the ludicrous configurability of PbNgen. Hat tip to [Daniel Munro](https://github.com/daniel-munro).

The matrix math that parallelizes the aforementioned stable ranking algorithm calculations making it performant even when running over hundreds of variously-shaped regions of color was developed with the help of [Gemini Alpha 2.5 Pro](https://gemini.google.com/). I would otherwise not have known how to [accomplish this](https://github.com/scottvr/pbngen/wiki/Paralellizing-the-Stable-Label-Locator).

The [mermaid](https://mermaidchart.com/) charts were generated for me by [ChatGPT](https://chatgpt.com/) and [Claude](https://claude.ai). Thank ya, boys!

-----

License
========

## [MIT](https://tlo.mit.edu/understand-ip/exploring-mit-open-source-license-comprehensive-guide)

Copyright © `2025` `scottvr @ gmail.com`

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
