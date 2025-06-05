# tests/test_file_utils.py
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
from pbn import file_utils


def test_save_pbn_png_creates_file_with_metadata(tmp_path):
    # Create a dummy image
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))

    output_file = tmp_path / "test_output.png"
    cmd_line = "pbngen --numcolors 3 dummy.png"
    metadata = {"User Note": "Test run", "Extra_Key": "Extra value"}

    file_utils.save_pbn_png(img, output_file, command_line_invocation=cmd_line, additional_metadata=metadata)

    # Validate file exists
    assert output_file.exists()

    # Validate PNG tEXt metadata is embedded
    with Image.open(output_file) as im:
        pnginfo = im.info
        assert "pbngen:command_line" in pnginfo
        assert "pbngen:User_Note" in pnginfo
        assert "pbngen:Extra_Key" in pnginfo
        assert pnginfo["pbngen:User_Note"] == "Test run"


def test_save_pbn_svg_creates_valid_svg(tmp_path):
    output_file = tmp_path / "test_output.svg"
    canvas_size = (100, 100)
    primitives = [
        {
            "outline": [[(10, 10), (90, 10), (90, 90), (10, 90), (10, 10)]],
            "labels": [{"value": 1, "position": (50, 50)}]
        }
    ]
    cmd_line = "pbngen --numcolors 3 dummy.png"
    metadata = {"Author": "Tester"}

    file_utils.save_pbn_svg(
        output_file, 
        canvas_size, 
        primitives, 
        command_line_invocation=cmd_line, 
        additional_metadata=metadata
    )

    # Validate SVG file exists
    assert output_file.exists()

    # Validate it can be parsed as XML
    tree = ET.parse(output_file)
    root = tree.getroot()
    assert root.tag.endswith("svg")


def test_verbatim_write_and_get_xml(tmp_path):
    xml_content = "<metadata><info>Test</info></metadata>"
    verbatim = file_utils.Verbatim(xml_string=xml_content, elementname="metadata")

    # Test write() method
    output_file = tmp_path / "verbatim_output.xml"
    with open(output_file, "w") as f:
        verbatim.write(f, indent=2)
    with open(output_file) as f:
        written_content = f.read()
    assert "<metadata><info>Test</info></metadata>" in written_content

    # Test get_xml() method
    element = verbatim.get_xml()
    assert isinstance(element, ET.Element)
    assert element.find("info").text == "Test"

