import os
import json
import numpy as np


def vec3_dict(v):
    return {
        "x": float(v[0]),
        "y": float(v[1]),
        "z": float(v[2]),
    }


def quat_dict(q):
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2]),
        "w": float(q[3]),
    }


def main():
    npz_path = "exports/motion_data/exported_motion.npz"
    meta_path = "exports/motion_data/exported_motion_meta.json"
    out_path = "exports/motion_data/exported_motion_to_unity.json"

    data = np.load(npz_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    root_states = data["root_states"]   # [T, 13]
    rb_pos = data["rb_pos"]             # [T, B, 3]
    rb_rot = data["rb_rot"]             # [T, B, 4]

    fps = float(meta["fps"]) if "fps" in meta else 1.0 / float(meta["dt"])
    body_names = meta["body_names"]

    num_frames = root_states.shape[0]
    num_bodies = len(body_names)

    unity_data = {
        "fps": fps,
        "num_frames": int(num_frames),
        "num_bodies": int(num_bodies),
        "body_names": body_names,
        "frames": []
    }

    for i in range(num_frames):
        frame = {
            "root_pos": vec3_dict(root_states[i][0:3]),
            "root_rot": quat_dict(root_states[i][3:7]),
            "body_pos": [vec3_dict(rb_pos[i][j]) for j in range(num_bodies)],
            "body_rot": [quat_dict(rb_rot[i][j]) for j in range(num_bodies)],
        }
        unity_data["frames"].append(frame)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(unity_data, f, indent=2)

    print(f"Saved Unity JSON to: {out_path}")
    print(f"Frames: {num_frames}")
    print(f"Bodies: {num_bodies}")
    print(f"FPS: {fps}")


if __name__ == "__main__":
    main()