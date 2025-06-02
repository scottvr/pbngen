import sys
from pathlib import Path
from PIL import Image, PngImagePlugin
from typing import Optional, Dict, Tuple, List
import svgwrite
import base64
import xml.etree.ElementTree as ET 
from xml.etree.ElementTree import Element, SubElement # Keep for clarity
from svgwrite.base import BaseElement # For Verbatim tag
import traceback
import re
import svgwrite.validator2 # For isinstance check

class Verbatim(BaseElement):
    def __init__(self, xml_string="", elementname="metadata", **kwargs_for_base_element):
        self.elementname = elementname # Instance attribute
        super(Verbatim, self).__init__(**kwargs_for_base_element)
        instance_el_name_after_super = getattr(self, 'elementname', 'ERROR: Instance elementname not set!')
        class_el_name = BaseElement.elementname
        validator_type_after_super = type(getattr(self, 'validator', None))
        self.xml_string = xml_string

    def write(self, fileobj, indent=0, newline='\n', options={}):
        current_elementname = getattr(self, 'elementname', 'ERROR: NOT SET IN WRITE')
        if current_elementname and not options.get('skip_validation', False) and hasattr(self, 'validator'):
            if not isinstance(self.validator, svgwrite.validator2.NullValidator):
                try:
                    self.validator._get_element(current_elementname)
                except KeyError as e:
                    raise
            else:
                print(f"DEBUG: Verbatim.write: Using NullValidator for '{current_elementname}', skipping _get_element call.")
        else:
            print(f"DEBUG: Verbatim.write: Skipping validation for '{current_elementname}'.")
        fileobj.write(self.xml_string)
        print(f"DEBUG: Verbatim.write for '{current_elementname}' completed.")

    def get_xml(self):
        # This method is now critical for the dwg.save(pretty=True) -> tostring() path.
        # It MUST return an xml.etree.ElementTree.Element if the caller uses ET.Element.append().
        current_elementname = getattr(self, 'elementname', 'ERROR: NOT SET IN get_xml')
        try:
            # self.xml_string is a complete XML block (e.g., "<metadata>...</metadata>")
            # ET.fromstring() parses an XML string and returns its root Element.
            et_element = ET.fromstring(self.xml_string)
            return et_element
        except ET.ParseError as e:
            print(f"ERROR in Verbatim.get_xml(): Failed to parse self.xml_string into an ET.Element. "
                  f"String was: '{self.xml_string}'. Error: {e}")
            # Re-raising the error is usually best so it's not silently ignored.
            raise

def save_pbn_png(
    image_to_save: Image.Image,
    output_path: Path, # Expecting a Path object
    command_line_invocation: Optional[str] = None,
    additional_metadata: Optional[Dict[str, str]] = None
):
    """
    Saves a PIL Image object as a PNG file, embedding specified metadata.
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path) # Ensure it's a Path object

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    png_info = PngImagePlugin.PngInfo()

    if command_line_invocation:
        png_info.add_text("pbngen:command_line", command_line_invocation)

    # Standard PBNG metadata
    png_info.add_text("Software", "pbngen by Scott VR (https://github.com/scottvr/pbngen)")
    
    if additional_metadata:
        for key, value in additional_metadata.items():
            # Simple key cleaning: replace spaces with underscores, ensure alphanumeric start
            key_clean = re.sub(r'\s+', '_', key)
            key_clean = re.sub(r'[^a-zA-Z0-9_.-]', '', key_clean)
            if not re.match(r'^[a-zA-Z_]', key_clean): # Must start with letter or underscore
                key_clean = "pbngen_" + key_clean

            # Ensure key is not overly long or problematic for tEXt chunks
            key_clean = key_clean[:70] # PNG tEXt chunk keyword limit is 79 bytes, leave room for prefix
            
            # Value should be string
            value_str = str(value)
            
            png_info.add_text(f"pbngen:{key_clean}", value_str)


    try:
        image_to_save.save(output_path, "PNG", pnginfo=png_info)
    except Exception as e:
        # Provide more context in the error message
        print(f"Error saving PNG to {output_path.resolve()}: {e}")
        # Consider re-raising or handling more gracefully depending on application needs
        # raise


def save_pbn_svg(
    output_path: Path, 
    canvas_size: Tuple[int, int],
    primitives: List[dict],
    font_path_str: Optional[str] = None,
    default_font_size: Optional[int] = 10,
    label_color_str: Optional[str] = "#88ddff", 
    outline_color_hex: str = "#88ddff", 
    command_line_invocation: Optional[str] = None,
    additional_metadata: Optional[dict[str, str]] = None
):
    if not isinstance(output_path, Path):
        output_path = Path(output_path) 

    width, height = canvas_size
    pbngen_ns_uri_for_custom_tags = 'http://www.github.com/scottvr/pbngen/assets/ns/pbngen#' 
    
    # --- Register namespace with ElementTree for cleaner output ---
    ET.register_namespace('pbngen', pbngen_ns_uri_for_custom_tags)
    
    dwg = svgwrite.Drawing(
        filename=str(output_path), 
        size=(f"{width}px", f"{height}px"),
        profile='full' 
    )

    # --- Font embedding logic ---
    font_family_svg = "sans-serif" 
    if font_path_str:
        try:
            font_path_obj = Path(font_path_str)
            if font_path_obj.is_file() and font_path_obj.suffix.lower() in ['.ttf', '.otf', '.woff', '.woff2']:
                font_family_svg = font_path_obj.stem 
                try:
                    with open(font_path_obj, 'rb') as f_font:
                        font_data = f_font.read()
                    font_data_b64 = base64.b64encode(font_data).decode('utf-8')
                    
                    font_mime_type = "font/ttf" 
                    if font_path_obj.suffix.lower() == ".otf": font_mime_type = "font/otf"
                    elif font_path_obj.suffix.lower() == ".woff": font_mime_type = "application/font-woff"
                    elif font_path_obj.suffix.lower() == ".woff2": font_mime_type = "font/woff2"
                    
                    font_face_css = f"@font-face {{"
                    font_face_css += f"font-family: '{font_family_svg}';"
                    font_face_css += f"src: url(data:{font_mime_type};base64,{font_data_b64});"
                    font_face_css += f"}}"
                    dwg.defs.add(dwg.style(content=font_face_css))
                except Exception as e:
                    print(f"Warning: Could not read or embed font {font_path_str} into SVG: {e}")
            else:
                print(f"Warning: Font path '{font_path_str}' is not a valid font file or unsupported type for SVG embedding.")
        except Exception as e: 
            print(f"Warning: Error processing font path '{font_path_str}' for SVG: {e}")
    # --- End Font embedding logic ---

    # --- Metadata block ---
    # Construct the entire <metadata> block using ElementTree
    
    # 1. Create the standard SVG <metadata> root element.
    #    No namespace prefix is needed here if it's to be the default <metadata> tag from SVG spec.
    #    ElementTree will handle adding xmlns for "pbngen" when pbngen-prefixed elements are added.
    std_metadata_root_et = Element('metadata') 
    std_metadata_root_et.set('id', 'pbngenApplicationDataContainer') # Optional ID for the container

    # 2. Create your custom namespaced <pbngen:pbngenMetadata> element as a child
    custom_metadata_et = SubElement(std_metadata_root_et, f'{{{pbngen_ns_uri_for_custom_tags}}}pbngenMetadata')
    custom_metadata_et.set('id', 'pbngenApplicationMetadata')

    # 3. Add your specific metadata items into the custom_metadata_et element
    software_el = SubElement(custom_metadata_et, f'{{{pbngen_ns_uri_for_custom_tags}}}Software')
    software_el.text = "pbngen by Scott Vardy (https://github.com/scottvr/pbngen)"

    if command_line_invocation:
        cli_el = SubElement(custom_metadata_et, f'{{{pbngen_ns_uri_for_custom_tags}}}CommandLineInvocation')
        cli_el.text = command_line_invocation
    
    if additional_metadata:
        for key, value in additional_metadata.items():
            el_name_local = re.sub(r'\s+', '_', key) 
            el_name_local = re.sub(r'[^a-zA-Z0-9_.-]', '', el_name_local) 
            if not el_name_local or not (el_name_local[0].isalpha() or el_name_local[0] == '_'):
                el_name_local = 'pbngen_' + el_name_local 
            
            meta_item_el = SubElement(custom_metadata_et, f'{{{pbngen_ns_uri_for_custom_tags}}}{el_name_local}')
            meta_item_el.text = str(value) 
    
    # 4. Serialize the complete std_metadata_root_et to an XML string
    #    This string will now start with <metadata id="..."> and contain <pbngen:pbngenMetadata xmlns:pbngen="...">...</pbngen:pbngenMetadata>
    full_metadata_xml_string = ET.tostring(std_metadata_root_et, encoding='unicode', method='xml')

    verbatim_metadata_obj = Verbatim(
        xml_string=full_metadata_xml_string,
        elementname='metadata',  # This is the named argument for Verbatim.__init__
        # These will be collected into **kwargs_for_base_element:
        profile=dwg.profile,
        debug=dwg.debug
    )
    dwg.add(verbatim_metadata_obj)

    # --- Drawing primitives ---
    outline_group = dwg.g(id="pbn-outlines", style=f"stroke:{outline_color_hex}; fill:none; stroke-width:1px;")
    for item in primitives:
        for contour_idx, contour in enumerate(item.get("outline", [])):
            filtered_points = [(max(0, min(int(round(x)), width)), max(0, min(int(round(y)), height))) for x, y in contour]
            if len(filtered_points) > 1:
                outline_group.add(dwg.polyline(points=filtered_points))
    dwg.add(outline_group)

    label_group = dwg.g(id="pbn-labels", style=f"fill:{label_color_str}; text-anchor:middle; alignment-baseline:middle; font-family:'{font_family_svg}', {'' if font_family_svg == 'sans-serif' else 'sans-serif'};") 
    for item in primitives:
        for label_idx, label in enumerate(item.get("labels", [])):
            x, y = label["position"]
            font_size_label = label.get("font_size", default_font_size if default_font_size else 10)
            label_group.add(dwg.text(str(label["value"]), insert=(int(round(x)), int(round(y))), 
                                     font_size=f"{font_size_label}px"
                                     ))
    dwg.add(label_group)
    # --- End Drawing primitives ---
    
    try:
        dwg.save(pretty=True) 
    except Exception as e:
        print(f"Error saving SVG to {output_path.resolve()}: {e}")
        print("--- Full Stack Trace Follows ---")
        traceback.print_exc() # This will print the full stack trace
        print("---------------------------------")