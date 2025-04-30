import svgwrite
from typing import List, Tuple

def write_svg(primitives: List[dict], output_path: str, canvas_size: Tuple[int, int]):
    """
    Render collected drawing primitives into an SVG file.

    Args:
        primitives (List[dict]): Drawing instructions with outlines and label positions.
        output_path (str): Path to save the SVG file.
        canvas_size (Tuple[int, int]): (width, height) of the SVG canvas.
    """
    dwg = svgwrite.Drawing(output_path, size=(canvas_size[0], canvas_size[1]))
    # Auto scale stroke width for visibility
    max_dim = max(canvas_size)
    stroke_width = max(0.5, int(max_dim / 512))  # tweak factor as needed


    for item in primitives:
        color = svgwrite.rgb(*item["color"])

        # Draw outline as polyline
        if item["outline"]:
#            dwg.add(dwg.polyline(points=item["outline"], stroke=color, fill="none", stroke_width=1))
            dwg.add(dwg.polyline(points=item["outline"], stroke=color, fill="none", stroke_width=stroke_width))


        # Add numeric labels
        for label in item["labels"]:
            x, y = label["position"]
            dwg.add(dwg.text(
                label["value"],
                insert=(x, y),
                fill=color,
                font_size=label["font_size"],
                text_anchor="middle",
                alignment_baseline="middle"
            ))

    dwg.save()
