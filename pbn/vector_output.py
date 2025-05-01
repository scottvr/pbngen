import svgwrite
from typing import Tuple
from .segment import collect_region_primitives

def write_svg(
    output_path: str,
    canvas_size: Tuple[int, int],
    primitives
):
    """
    Render collected drawing primitives into an SVG file with optional debugging.
    """
    width, height = canvas_size
    dwg = svgwrite.Drawing(output_path, size=(width, height))

    # Optional debug box to visualize canvas bounds
    dwg.add(dwg.rect(insert=(0, 0), size=canvas_size, fill='white', stroke='gray', stroke_width=1))

    for item in primitives:
#        try:
#            color = svgwrite.rgb(*item["color"])
#        except Exception as e:
#            raise ValueError(f"Invalid color value in primitive: {item['color']}") from e
        color = "#66ccff"  # fixed light blue for PBN outlines and labels


        # Draw outlines
        if item.get("outline"):
            for contour in item["outline"]:
                filtered = []
                for x, y in contour:
                    if 0 <= x < width and 0 <= y < height:
                        filtered.append((int(x), int(y)))
                    else:
                        print(f"⚠️ Outline point ({x},{y}) out of bounds")

                if filtered:
                    dwg.add(dwg.polyline(
                        points=filtered,
                        stroke=color,
                        fill="none",
                        stroke_width=1
                    ))

        # Draw text labels
        for label in item.get("labels", []):
            x, y = label["position"]
            if not (0 <= x < width and 0 <= y < height):
                print(f"⚠️ Label '{label['value']}' at ({x},{y}) is out of bounds")
                continue

            dwg.add(dwg.text(
                label["value"],
                insert=(int(x), int(y)),
                fill=color,
                font_size=label["font_size"],
                text_anchor="middle",
                alignment_baseline="middle"
            ))

    dwg.save()
