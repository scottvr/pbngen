#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# Define the namespace URI used in your SVGs and the prefix for PNG metadata
PBNGEN_SVG_NS_URI = 'http://www.github.com/scottvr/pbngen/assets/ns/pbngen#' #
PNG_METADATA_PREFIX = "pbngen:" #

def extract_png_metadata(filepath: Path):
    """
    Extracts and prints PbNgen metadata from a PNG file.
    """
    print(f"--- PbNgen Metadata for PNG: {filepath.name} ---")
    try:
        with Image.open(filepath) as img:
            # Metadata is typically in img.info (which often includes .text items)
            # The keys are exactly as written by save_pbn_png
            found_metadata = False
            if img.info:
                for key, value in img.info.items():
                    if key.startswith(PNG_METADATA_PREFIX):
                        # Clean up the key for display (remove prefix)
                        display_key = key[len(PNG_METADATA_PREFIX):]
                        print(f"  {display_key}: {value}")
                        found_metadata = True
            
            if not found_metadata:
                print("  No PbNgen-specific metadata found.")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error processing PNG file {filepath}: {e}")
    print("-" * (30 + len(filepath.name)))


def extract_svg_metadata(filepath: Path):
    """
    Extracts and prints PbNgen metadata from an SVG file.
    """
    print(f"--- PbNgen Metadata for SVG: {filepath.name} ---")
    
    SVG_NS = 'http://www.w3.org/2000/svg'  # Standard SVG Namespace

    try:
        tree = ET.parse(filepath)
        root = tree.getroot() # This is the <svg> element
        
        metadata_element = None
        
        # Look for the <metadata> element (which should be in the SVG namespace)
        # as a direct child of the root <svg> element.
        for child in root: # Iterate direct children of <svg>
            if child.tag == f'{{{SVG_NS}}}metadata' and child.get('id') == 'pbngenApplicationMetadata':
                metadata_element = child
                break
        
        # Fallback: If not found with specific ID, find the first svg:metadata
        if metadata_element is None:
            for child in root:
                if child.tag == f'{{{SVG_NS}}}metadata':
                    metadata_element = child
                    print(f"  Note: Found a general <metadata> block (not id='pbngenApplicationMetadata'). Inspecting its PbNgen content...")
                    break # Use the first one found

        if metadata_element is not None:
            found_pbngen_metadata = False
            for custom_elem in metadata_element:
                if custom_elem.tag.startswith(f'{{{PBNGEN_SVG_NS_URI}}}'):
                    local_name = custom_elem.tag.split('}', 1)[1]
                    print(f"  {local_name}: {custom_elem.text.strip() if custom_elem.text else ''}")
                    found_pbngen_metadata = True
            
            if not found_pbngen_metadata:
                print(f"  The <metadata> block was found, but it contains no PbNgen-specific elements in the '{PBNGEN_SVG_NS_URI}' namespace.")
        else:
            print(f"  No <metadata> block (id='pbngenApplicationMetadata' or any standard svg:metadata) found in '{filepath.name}'.") # Updated message

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except ET.ParseError:
        print(f"Error: Could not parse SVG file (invalid XML): {filepath}")
    except Exception as e:
        print(f"Error processing SVG file {filepath}: {e}")
    print("-" * (30 + len(filepath.name)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_pbngen_meta.py <filename.png_or_svg>")
        sys.exit(1)

    filepath_str = sys.argv[1]
    filepath = Path(filepath_str)

    if not filepath.is_file():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    file_extension = filepath.suffix.lower()

    if file_extension == ".png":
        extract_png_metadata(filepath)
    elif file_extension == ".svg":
        extract_svg_metadata(filepath)
    else:
        print(f"Error: Unsupported file type '{file_extension}'. Please provide a .png or .svg file.")
        sys.exit(1)

if __name__ == "__main__":
    main()