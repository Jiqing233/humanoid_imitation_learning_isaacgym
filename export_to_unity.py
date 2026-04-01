import os
import json
import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    return quat_conjugate(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiply for xyzw format.
    Returns q = q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return normalize_quat(np.array([x, y, z, w], dtype=np.float64))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q, all in xyzw convention.
    """
    q = normalize_quat(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return quat_multiply(quat_multiply(q, vq), quat_inverse(q))[:3]


def vec3_dict(v):
    return {
        "x": float(v[0]),
        "y": float(v[1]),
        "z": float(v[2]),
    }


def quat_dict(q):
    q = normalize_quat(q)
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2]),
        "w": float(q[3]),
    }


# ============================================================
# Coordinate conversion hooks
# ============================================================
# Right now these are identity transforms.
# If your avatar appears mirrored / lying down / twisted,
# this is one of the first places to adjust.
# ============================================================

def convert_position_to_unity(p: np.ndarray) -> np.ndarray:
    """
    Isaac -> Unity position conversion hook.
    Currently identity.
    """
    return np.asarray(p, dtype=np.float64)


def convert_quaternion_to_unity(q: np.ndarray) -> np.ndarray:
    """
    Isaac -> Unity quaternion conversion hook.
    Currently identity.
    """
    return normalize_quat(np.asarray(q, dtype=np.float64))


# ============================================================
# Local transform conversion
# ============================================================

def compute_local_transform(
    parent_pos: np.ndarray,
    parent_rot: np.ndarray,
    child_pos: np.ndarray,
    child_rot: np.ndarray,
):
    """
    Convert child global transform into parent-relative local transform.

    local_rot = inv(parent_rot) * child_rot
    local_pos = inv(parent_rot) * (child_pos - parent_pos)
    """
    parent_rot_inv = quat_inverse(parent_rot)

    local_rot = quat_multiply(parent_rot_inv, child_rot)
    local_pos = quat_rotate(parent_rot_inv, child_pos - parent_pos)

    return local_pos, local_rot


def sanitize_parent_indices(body_names, parent_indices):
    """
    Validate hierarchy length and values.
    """
    if len(body_names) != len(parent_indices):
        raise ValueError(
            f"Hierarchy mismatch: len(body_names)={len(body_names)} "
            f"but len(parent_indices)={len(parent_indices)}"
        )

    n = len(body_names)
    for i, p in enumerate(parent_indices):
        if p < -1 or p >= n:
            raise ValueError(f"Invalid parent index for body {i} ({body_names[i]}): {p}")
        if p == i:
            raise ValueError(f"Body {i} ({body_names[i]}) cannot be its own parent.")


# ============================================================
# Main export
# ============================================================

def main():
    npz_path = "exports/motion_data/exported_motion.npz"
    meta_path = "exports/motion_data/exported_motion_meta.json"
    out_path = "exports/motion_data/exported_motion_to_unity_local.json"

    data = np.load(npz_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    root_states = data["root_states"]   # [T, 13]
    rb_pos = data["rb_pos"]             # [T, B, 3]
    rb_rot = data["rb_rot"]             # [T, B, 4]

    fps = float(meta["fps"]) if "fps" in meta else 1.0
    body_names = list(meta["body_names"])
    parent_indices = list(meta["body_parent"])

    sanitize_parent_indices(body_names, parent_indices)

    num_frames = int(root_states.shape[0])
    num_bodies = int(len(body_names))

    # --------------------------------------------------------
    # Convert globals into Unity basis first
    # --------------------------------------------------------
    rb_pos_u = np.zeros_like(rb_pos, dtype=np.float64)
    rb_rot_u = np.zeros_like(rb_rot, dtype=np.float64)

    for t in range(num_frames):
        for b in range(num_bodies):
            rb_pos_u[t, b] = convert_position_to_unity(rb_pos[t, b])
            rb_rot_u[t, b] = convert_quaternion_to_unity(rb_rot[t, b])

    root_pos_u = np.zeros((num_frames, 3), dtype=np.float64)
    root_rot_u = np.zeros((num_frames, 4), dtype=np.float64)

    for t in range(num_frames):
        root_pos_u[t] = convert_position_to_unity(root_states[t, 0:3])
        root_rot_u[t] = convert_quaternion_to_unity(root_states[t, 3:7])

    # --------------------------------------------------------
    # Compute bind local positions from frame 0
    # --------------------------------------------------------
    bind_local_pos = []
    bind_local_rot = []

    for b in range(num_bodies):
        p = parent_indices[b]
        if p == -1:
            # Root body in the skeleton hierarchy
            local_pos = rb_pos_u[0, b]
            local_rot = rb_rot_u[0, b]
        else:
            local_pos, local_rot = compute_local_transform(
                rb_pos_u[0, p], rb_rot_u[0, p],
                rb_pos_u[0, b], rb_rot_u[0, b]
            )

        bind_local_pos.append(vec3_dict(local_pos))
        bind_local_rot.append(quat_dict(local_rot))

    # --------------------------------------------------------
    # Build per-frame local rotations
    # Keep root motion from root_states separately
    # --------------------------------------------------------
    frames = []

    for t in range(num_frames):
        local_rots = []
        local_pos_debug = []

        for b in range(num_bodies):
            p = parent_indices[b]
            if p == -1:
                # For the skeleton root body, local == global in skeleton space.
                local_pos = rb_pos_u[t, b]
                local_rot = rb_rot_u[t, b]
            else:
                local_pos, local_rot = compute_local_transform(
                    rb_pos_u[t, p], rb_rot_u[t, p],
                    rb_pos_u[t, b], rb_rot_u[t, b]
                )

            local_rots.append(quat_dict(local_rot))
            local_pos_debug.append(vec3_dict(local_pos))

        frame = {
            "root_pos": vec3_dict(root_pos_u[t]),
            "root_rot": quat_dict(root_rot_u[t]),
            "local_rot": local_rots,

            # Debug only:
            # useful for checking whether local positions stay stable over time
            "local_pos_debug": local_pos_debug,
        }
        frames.append(frame)

    unity_data = {
        "format": "isaac_to_unity_local_skeleton_v1",
        "fps": fps,
        "num_frames": num_frames,
        "num_bodies": num_bodies,
        "body_names": body_names,
        "parent_indices": parent_indices,

        # Use these as bind/rest local offsets on the Unity side.
        "bind_local_pos": bind_local_pos,
        "bind_local_rot": bind_local_rot,

        # Helpful metadata for debugging
        "notes": {
            "quaternion_order": "xyzw",
            "root_motion_in_world_space": True,
            "bone_rotations_are_parent_relative": True,
            "bind_local_pos_from_frame0": True,
            "coordinate_conversion_applied": "identity",
        },

        "frames": frames,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(unity_data, f, indent=2)

    print(f"Saved Unity local-animation JSON to: {out_path}")
    print(f"Frames: {num_frames}")
    print(f"Bodies: {num_bodies}")
    print(f"FPS: {fps}")


if __name__ == "__main__":
    main()