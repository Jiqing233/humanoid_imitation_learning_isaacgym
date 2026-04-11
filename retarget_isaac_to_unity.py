import os
import json
import math
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def vec3_from_dict(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float32)


def quat_from_dict(d):
    q = np.array([d["x"], d["y"], d["z"], d["w"]], dtype=np.float32)
    return quat_normalize(q)


def vec3_to_dict(v):
    return {
        "x": float(v[0]),
        "y": float(v[1]),
        "z": float(v[2]),
    }


def quat_to_dict(q):
    q = quat_normalize(q)
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2]),
        "w": float(q[3]),
    }


def quat_identity():
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return quat_identity()
    return q / n


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_inverse(q):
    q = quat_normalize(q)
    return quat_conjugate(q)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return quat_normalize(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=np.float32))


def quat_slerp(q0, q1, t):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return quat_normalize(out)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t

    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return quat_normalize((s0 * q0) + (s1 * q1))


def quat_pow(q, weight):
    """
    Interpolate from identity to q by 'weight'.
    Equivalent to a simple fractional rotation.
    """
    return quat_slerp(quat_identity(), quat_normalize(q), float(weight))


def build_name_to_index(names):
    return {name: i for i, name in enumerate(names)}


def get_config_quat(config, key, default=None):
    if key not in config:
        return quat_identity() if default is None else default
    arr = config[key]
    return quat_normalize(np.array(arr, dtype=np.float32))


def main():
    # ------------------- config -------------------
    config_path = "configs/isaac_to_banana.json"
    # ------------------------------------------------

    config = load_json(config_path)

    source_path = config["source_clip"]
    target_path = config["target_skeleton"]
    output_path = config["output_clip"]

    source = load_json(source_path)
    target = load_json(target_path)

    source_bone_names = source["bone_names"]
    target_bone_names = target["bone_names"]

    source_name_to_idx = build_name_to_index(source_bone_names)
    target_name_to_idx = build_name_to_index(target_bone_names)

    source_bind_local_rot = [
        quat_from_dict(q) for q in source["source_bind_local_rot"]
    ]
    target_bind_local_rot = [
        quat_from_dict(q) for q in target["target_bind_local_rot"]
    ]

    source_bind_local_pos = [
        vec3_from_dict(v) for v in source["source_bind_local_pos"]
    ]
    target_bind_local_pos = [
        vec3_from_dict(v) for v in target["target_bind_local_pos"]
    ]

    direct_map = config.get("direct_map", {})
    chain_map = config.get("chain_map", {})
    distribution_weights = config.get("distribution_weights", {})
    bone_corrections_raw = config.get("bone_corrections_xyzw", {})

    root_scale = float(config.get("root_scale", 1.0))
    root_offset = np.array(config.get("root_offset", [0.0, 0.0, 0.0]), dtype=np.float32)

    bone_corrections = {}
    for tgt_name, q_list in bone_corrections_raw.items():
        bone_corrections[tgt_name] = quat_normalize(np.array(q_list, dtype=np.float32))

    # Optional: initialize all target bones to bind pose per frame
    num_target_bones = len(target_bone_names)

    output_frames = []

    for frame_idx, src_frame in enumerate(source["frames"]):
        target_local_rot = [q.copy() for q in target_bind_local_rot]

        # ---------------- root position ----------------
        src_root_pos = vec3_from_dict(src_frame["root_pos"])
        out_root_pos = src_root_pos * root_scale + root_offset

        # ---------------- source locals ----------------
        src_local_rots = [quat_from_dict(q) for q in src_frame["source_local_rot"]]

        # ---------------- direct one-to-one mapping ----------------
        for src_name, tgt_name in direct_map.items():
            if src_name not in source_name_to_idx:
                print(f"[WARN] direct_map source bone not found: {src_name}")
                continue
            if tgt_name not in target_name_to_idx:
                print(f"[WARN] direct_map target bone not found: {tgt_name}")
                continue

            s_idx = source_name_to_idx[src_name]
            t_idx = target_name_to_idx[tgt_name]

            src_bind = source_bind_local_rot[s_idx]
            src_cur = src_local_rots[s_idx]
            src_delta = quat_mul(quat_inverse(src_bind), src_cur)

            correction = bone_corrections.get(tgt_name, quat_identity())
            tgt_bind = target_bind_local_rot[t_idx]

            tgt_local = quat_mul(quat_mul(tgt_bind, correction), src_delta)
            target_local_rot[t_idx] = quat_normalize(tgt_local)

        # ---------------- one-to-many chain mapping ----------------
        for src_name, tgt_list in chain_map.items():
            if src_name not in source_name_to_idx:
                print(f"[WARN] chain_map source bone not found: {src_name}")
                continue

            missing_targets = [t for t in tgt_list if t not in target_name_to_idx]
            if missing_targets:
                print(f"[WARN] chain_map target bones not found for {src_name}: {missing_targets}")
                continue

            weights = distribution_weights.get(src_name, None)
            if weights is None:
                # equal split fallback
                weights = [1.0 / len(tgt_list)] * len(tgt_list)

            if len(weights) != len(tgt_list):
                raise ValueError(
                    f"distribution_weights for '{src_name}' has length {len(weights)}, "
                    f"but chain_map has length {len(tgt_list)}"
                )

            s_idx = source_name_to_idx[src_name]
            src_bind = source_bind_local_rot[s_idx]
            src_cur = src_local_rots[s_idx]
            src_delta = quat_mul(quat_inverse(src_bind), src_cur)

            for tgt_name, w in zip(tgt_list, weights):
                t_idx = target_name_to_idx[tgt_name]
                tgt_bind = target_bind_local_rot[t_idx]
                correction = bone_corrections.get(tgt_name, quat_identity())

                partial_delta = quat_pow(src_delta, w)
                tgt_local = quat_mul(quat_mul(tgt_bind, correction), partial_delta)
                target_local_rot[t_idx] = quat_normalize(tgt_local)

        # ---------------- build output frame ----------------
        out_frame = {
            "root_pos": vec3_to_dict(out_root_pos),
            "target_local_rot": [quat_to_dict(q) for q in target_local_rot]
        }
        output_frames.append(out_frame)

    output = {
        "target_name": target["target_name"],
        "bone_names": target["bone_names"],
        "parent_indices": target["parent_indices"],
        "target_bind_local_pos": target["target_bind_local_pos"],
        "target_bind_local_rot": target["target_bind_local_rot"],
        "frames": output_frames
    }

    save_json(output_path, output)
    print(f"[INFO] Retargeted clip saved to: {output_path}")
    print(f"[INFO] Source clip: {source_path}")
    print(f"[INFO] Target skeleton: {target_path}")
    print(f"[INFO] Output frames: {len(output_frames)}")


if __name__ == "__main__":
    main()