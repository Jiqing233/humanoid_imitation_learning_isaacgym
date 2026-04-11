#!/usr/bin/env python3
"""
Extract the Isaac Gym humanoid skeleton from the MuJoCo XML.
No Isaac Gym / GPU dependency required.

Output: exports/isaac_static_skeleton.json
"""
import json
import math
import os
import xml.etree.ElementTree as ET


def main():
    xml_path = "assets/humanoid.xml"
    output_path = "exports/isaac_static_skeleton.json"

    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")

    bone_names = []
    parent_indices = []
    local_positions = []

    def walk(elem, parent_idx):
        idx = len(bone_names)
        bone_names.append(elem.get("name"))
        parent_indices.append(parent_idx)
        pos = [float(x) for x in elem.get("pos", "0 0 0").split()]
        local_positions.append({"x": pos[0], "y": pos[1], "z": pos[2]})
        for child in elem.findall("body"):
            walk(child, idx)

    for body in worldbody.findall("body"):
        walk(body, -1)

    # A-pose: rotate upper arms +-45 deg around X
    half = math.radians(45.0) / 2.0
    s, c = math.sin(half), math.cos(half)
    name_to_idx = {n: i for i, n in enumerate(bone_names)}

    a_pose_rot = [{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0} for _ in bone_names]
    if "right_upper_arm" in name_to_idx:
        a_pose_rot[name_to_idx["right_upper_arm"]] = {
            "x": float(-s), "y": 0.0, "z": 0.0, "w": float(c)
        }
    if "left_upper_arm" in name_to_idx:
        a_pose_rot[name_to_idx["left_upper_arm"]] = {
            "x": float(s), "y": 0.0, "z": 0.0, "w": float(c)
        }

    skeleton = {
        "source_name": "isaac_humanoid",
        "bone_names": bone_names,
        "parent_indices": parent_indices,
        "source_bind_local_pos": local_positions,
        "source_bind_local_rot": a_pose_rot,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(skeleton, f, indent=2)

    print(f"Extracted {len(bone_names)} bones: {bone_names}")
    print(f"Parent indices: {parent_indices}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
