import xml.etree.ElementTree as ET


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []

    for object in root.findall("object"):
        xml_info = {}

        type_tag = object.find("name")
        xml_info["class"] = type_tag.text

        bbox_tag = object.find("bndbox")
        bbox_info = {"xmin": int(bbox_tag.find("xmin").text),
                     "xmax": int(bbox_tag.find("xmax").text),
                     "ymin": int(bbox_tag.find("ymin").text),
                     "ymax": int(bbox_tag.find("ymax").text)}

        xml_info["bbox"] = bbox_info

        objects.append(xml_info)

    return objects
