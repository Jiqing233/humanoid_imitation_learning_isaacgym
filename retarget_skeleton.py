#!/usr/bin/env python3
"""
Retarget Isaac Gym skeleton to Unity avatar skeleton.

Fixes over the previous retarget_isaac_to_unity.py:
1. Proper coordinate system conversion (Isaac Z-up right-handed -> Unity Y-up left-handed)
2. World-space rotation deltas (not local-space, which breaks with different bone orientations)
3. Cumulative weights for chain mapping (ensures total rotation at chain end matches source)

Coordinate systems:
    Isaac (MuJoCo): X=forward, Y=left,  Z=up      (right-handed)
    Unity:          X=right,   Y=up,    Z=forward  (left-handed)

    Position conversion:   Unity = (-Isaac.y,  Isaac.z, Isaac.x)
    Quaternion conversion: Unity = ( Isaac.qy, -Isaac.qz, -Isaac.qx, Isaac.qw)

Usage:
    python retarget_skeleton.py                       # retarget source_clip.json animation
    python retarget_skeleton.py --test                # generate test poses for debugging
    python retarget_skeleton.py --config path.json    # use custom config
"""
import json
import math
import os
import sys
import numpy as np


# ============================================================
# Quaternion math (XYZW order)
# ============================================================

def quat_identity():
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else quat_identity()


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_inverse(q):
    return quat_conjugate(quat_normalize(q))


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return quat_normalize(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]))


def quat_rotate(q, v):
    """Rotate 3D vector v by unit quaternion q. Preserves vector magnitude.
    Uses the efficient formula: result = v + 2*w*(u x v) + 2*(u x (u x v))
    where q = (ux, uy, uz, w).
    """
    u = q[:3]
    w = q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_slerp(q0, q1, t):
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta = math.acos(min(1.0, max(-1.0, dot)))
    st = math.sin(theta)
    return quat_normalize(
        q0 * math.sin((1 - t) * theta) / st +
        q1 * math.sin(t * theta) / st
    )


def quat_pow(q, t):
    """Fractional rotation: slerp from identity to q by amount t."""
    return quat_slerp(quat_identity(), quat_normalize(q), float(t))


def quat_angle_deg(q):
    """Rotation angle in degrees."""
    q = quat_normalize(q)
    return 2.0 * math.degrees(math.acos(min(1.0, max(-1.0, abs(q[3])))))


# ============================================================
# Coordinate conversion: Isaac <-> Unity
#
# Derivation:
#   Isaac basis: e1=forward, e2=left, e3=up  (right-handed: e1 x e2 = e3)
#   Unity basis: f1=right,  f2=up,  f3=fwd   (left-handed)
#
#   Mapping: e1->f3, e2->-f1, e3->f2
#   Transform matrix C (det=-1, includes handedness flip):
#     [ 0 -1  0]
#     [ 0  0  1]
#     [ 1  0  0]
#
#   Position:   p_unity = C * p_isaac = (-py, pz, px)
#   Quaternion: Since det(C)=-1, the rotation direction flips for reflected axes.
#               q_unity = (qy, -qz, -qx, qw)
# ============================================================

def isaac_pos_to_unity(p):
    return np.array([-p[1], p[2], p[0]])


def isaac_quat_to_unity(q):
    return quat_normalize(np.array([q[1], -q[2], -q[0], q[3]]))


# ============================================================
# I/O
# ============================================================

def v3(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)


def q4(d):
    return quat_normalize(np.array([d["x"], d["y"], d["z"], d["w"]], dtype=np.float64))


def v3d(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def q4d(q):
    q = quat_normalize(q)
    return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Forward kinematics
# ============================================================

def fk_rotations(parents, local_rot):
    """FK: local rotations -> world rotations."""
    n = len(parents)
    world = [None] * n
    for i in range(n):
        p = parents[i]
        world[i] = local_rot[i].copy() if p == -1 else quat_mul(world[p], local_rot[i])
    return world


def fk_full(parents, local_pos, local_rot):
    """FK: local pos+rot -> world pos+rot."""
    n = len(parents)
    wp, wr = [None] * n, [None] * n
    for i in range(n):
        p = parents[i]
        if p == -1:
            wp[i], wr[i] = local_pos[i].copy(), local_rot[i].copy()
        else:
            wr[i] = quat_mul(wr[p], local_rot[i])
            wp[i] = wp[p] + quat_rotate(wr[p], local_pos[i])
    return wp, wr


# ============================================================
# Retargeting
# ============================================================

def build_bone_mapping(config, src_names, tgt_names):
    """
    Build (src_idx, tgt_idx, cumulative_weight) tuples from config.

    For chain maps, weights are accumulated so that the last bone in the
    chain carries the full source delta (weight=1.0).  This ensures the
    world rotation at the chain's tip matches the source bone exactly,
    while intermediate bones share the rotation proportionally.
    """
    si = {n: i for i, n in enumerate(src_names)}
    ti = {n: i for i, n in enumerate(tgt_names)}
    mapping = []

    for s, t in config.get("direct_map", {}).items():
        if s in si and t in ti:
            mapping.append((si[s], ti[t], 1.0))
        else:
            print(f"  [WARN] direct_map bone not found: {s} -> {t}")

    for s, tgt_list in config.get("chain_map", {}).items():
        if s not in si:
            print(f"  [WARN] chain_map source not found: {s}")
            continue
        weights = config.get("distribution_weights", {}).get(s)
        if weights is None:
            weights = [1.0 / len(tgt_list)] * len(tgt_list)
        cum = 0.0
        for t, w in zip(tgt_list, weights):
            cum += w
            if t in ti:
                mapping.append((si[s], ti[t], cum))
            else:
                print(f"  [WARN] chain_map target not found: {t}")

    return mapping


def retarget_frame(src_parents, src_bind_local, src_cur_local,
                   tgt_parents, tgt_bind_local, bone_mapping):
    """
    Retarget one frame using aligned world-space rotation deltas.

    The source and target may have different root bind orientations (e.g.
    source root = identity while target root has a large X rotation from
    a Z-up model space).  A naive world-space delta would turn a source
    yaw into a target tumble.

    Fix: conjugate each world-space delta by the root alignment rotation
    so the delta is expressed in the target's reference frame.

      align          = src_bw[root] * inv(tgt_bw[root])
      conjugated_Δ   = inv(align) * Δ * align
                      = tgt_bw[root] * inv(src_bw[root]) * Δ * src_bw[root] * inv(tgt_bw[root])
      desired_world  = conjugated_Δ * tgt_bw[bone]

    When src_bw[root] = identity this simplifies to conjugation by tgt_bw[root].
    """
    n_tgt = len(tgt_parents)

    src_bw = fk_rotations(src_parents, src_bind_local)
    src_cw = fk_rotations(src_parents, src_cur_local)
    tgt_bw = fk_rotations(tgt_parents, tgt_bind_local)

    # Root alignment: conjugate deltas so a source yaw stays a target yaw
    align = quat_mul(src_bw[0], quat_inverse(tgt_bw[0]))
    inv_align = quat_inverse(align)

    # Fast lookup: target bone index -> (source index, cumulative weight)
    tgt_map = {}
    for si, ti, w in bone_mapping:
        tgt_map[ti] = (si, w)

    tgt_cl = [r.copy() for r in tgt_bind_local]   # output local rotations
    tgt_cw = [None] * n_tgt                        # running world rotations

    for i in range(n_tgt):
        p = tgt_parents[i]

        if i in tgt_map:
            sidx, w = tgt_map[i]
            delta = quat_mul(src_cw[sidx], quat_inverse(src_bw[sidx]))
            if w < 1.0 - 1e-6:
                delta = quat_pow(delta, w)
            # Conjugate delta into target's reference frame
            conj_delta = quat_mul(inv_align, quat_mul(delta, align))
            desired = quat_mul(conj_delta, tgt_bw[i])
            tgt_cw[i] = desired
            tgt_cl[i] = desired if p == -1 else quat_mul(quat_inverse(tgt_cw[p]), desired)
        else:
            tgt_cw[i] = tgt_cl[i].copy() if p == -1 else quat_mul(tgt_cw[p], tgt_cl[i])

    return tgt_cl


# ============================================================
# Main
# ============================================================

def main():
    # --- Parse CLI ---
    config_path = "configs/isaac_to_unity_Lee8.json"
    test_mode = "--test" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    config = load_json(config_path)
    target = load_json(config["target_skeleton"])

    # In test mode, use extracted static skeleton; otherwise use source_clip
    if test_mode:
        src_path = "exports/isaac_static_skeleton.json"
        if not os.path.exists(src_path):
            print(f"[ERROR] Run extract_isaac_skeleton.py first to create {src_path}")
            sys.exit(1)
    else:
        src_path = config["source_clip"]

    source = load_json(src_path)

    src_names = source["bone_names"]
    tgt_names = target["bone_names"]
    src_parents = source["parent_indices"]
    tgt_parents = target["parent_indices"]

    src_bind_pos = [v3(d) for d in source["source_bind_local_pos"]]
    src_bind_rot = [q4(d) for d in source["source_bind_local_rot"]]
    tgt_bind_pos = [v3(d) for d in target["target_bind_local_pos"]]
    tgt_bind_rot = [q4(d) for d in target["target_bind_local_rot"]]

    # Convert source skeleton to Unity coordinate system
    src_bind_pos_u = [isaac_pos_to_unity(p) for p in src_bind_pos]
    src_bind_rot_u = [isaac_quat_to_unity(q) for q in src_bind_rot]

    # Build mapping
    mapping = build_bone_mapping(config, src_names, tgt_names)

    print(f"Source: {len(src_names)} bones  ({src_path})")
    print(f"Target: {len(tgt_names)} bones  ({config['target_skeleton']})")
    print(f"Mappings ({len(mapping)}):")
    for si, ti, w in mapping:
        print(f"  {src_names[si]:20s} -> {tgt_names[ti]:20s}  cum_w={w:.2f}")

    # Height ratio for root position scaling
    _, _ = fk_full(src_parents, src_bind_pos_u, src_bind_rot_u)
    src_wp, _ = fk_full(src_parents, src_bind_pos_u, src_bind_rot_u)
    tgt_wp, _ = fk_full(tgt_parents, tgt_bind_pos, tgt_bind_rot)
    src_h = max(p[1] for p in src_wp) - min(p[1] for p in src_wp)
    tgt_h = max(p[1] for p in tgt_wp) - min(p[1] for p in tgt_wp)
    root_scale = tgt_h / src_h if src_h > 1e-6 else 1.0
    print(f"Source height (Unity coords): {src_h:.4f}")
    print(f"Target height:                {tgt_h:.4f}")
    print(f"Root scale:                   {root_scale:.6f}")

    # --- Generate frames ---
    frames = []

    if test_mode:
        print("\nGenerating test frames...")

        # Frame 0: Bind pose (delta=identity -> output should equal target bind)
        r0 = retarget_frame(src_parents, src_bind_rot_u, src_bind_rot_u,
                            tgt_parents, tgt_bind_rot, mapping)
        max_err = max(quat_angle_deg(quat_mul(quat_inverse(r0[i]), tgt_bind_rot[i]))
                      for i in range(len(tgt_names)))
        print(f"  Frame 0 (A-pose / bind):  max angular error = {max_err:.4f} deg")
        frames.append({"root_pos": v3d(tgt_wp[0]),
                        "target_local_rot": [q4d(q) for q in r0],
                        "label": "bind_pose"})

        # Frame 1: T-pose (all identity rotations in Isaac -> arms straight down)
        t_pose_isaac = [quat_identity() for _ in src_names]
        t_pose_u = [isaac_quat_to_unity(q) for q in t_pose_isaac]
        r1 = retarget_frame(src_parents, src_bind_rot_u, t_pose_u,
                            tgt_parents, tgt_bind_rot, mapping)
        frames.append({"root_pos": v3d(tgt_wp[0]),
                        "target_local_rot": [q4d(q) for q in r1],
                        "label": "t_pose"})
        print("  Frame 1 (T-pose / arms down)")

        # Frame 2: Right arm raised 45 deg from A-pose (rotate around Isaac Y axis)
        arm_cur = [q.copy() for q in src_bind_rot]
        arm_idx = {n: i for i, n in enumerate(src_names)}.get("right_upper_arm")
        if arm_idx is not None:
            half = math.radians(45.0) / 2.0
            extra = np.array([0, math.sin(half), 0, math.cos(half)])
            arm_cur[arm_idx] = quat_mul(extra, arm_cur[arm_idx])
        arm_cur_u = [isaac_quat_to_unity(q) for q in arm_cur]
        r2 = retarget_frame(src_parents, src_bind_rot_u, arm_cur_u,
                            tgt_parents, tgt_bind_rot, mapping)
        frames.append({"root_pos": v3d(tgt_wp[0]),
                        "target_local_rot": [q4d(q) for q in r2],
                        "label": "right_arm_raised_45"})
        print("  Frame 2 (right arm +45 deg)")

        # Frame 3: Torso rotated 30 deg around Isaac Z (yaw left)
        torso_cur = [q.copy() for q in src_bind_rot]
        torso_idx = {n: i for i, n in enumerate(src_names)}.get("torso")
        if torso_idx is not None:
            half = math.radians(30.0) / 2.0
            extra = np.array([0, 0, math.sin(half), math.cos(half)])
            torso_cur[torso_idx] = quat_mul(extra, torso_cur[torso_idx])
        torso_cur_u = [isaac_quat_to_unity(q) for q in torso_cur]
        r3 = retarget_frame(src_parents, src_bind_rot_u, torso_cur_u,
                            tgt_parents, tgt_bind_rot, mapping)
        frames.append({"root_pos": v3d(tgt_wp[0]),
                        "target_local_rot": [q4d(q) for q in r3],
                        "label": "torso_yaw_30"})
        print("  Frame 3 (torso yaw +30 deg)")

    else:
        # Retarget animation from source_clip
        src_frames = source.get("frames", [])
        print(f"\nRetargeting {len(src_frames)} animation frames...")
        for fi, sf in enumerate(src_frames):
            cur = [q4(d) for d in sf["source_local_rot"]]
            cur_u = [isaac_quat_to_unity(q) for q in cur]
            rl = retarget_frame(src_parents, src_bind_rot_u, cur_u,
                                tgt_parents, tgt_bind_rot, mapping)
            rp = isaac_pos_to_unity(v3(sf["root_pos"])) * root_scale
            frames.append({"root_pos": v3d(rp),
                            "target_local_rot": [q4d(q) for q in rl]})
        print(f"  Retargeted {len(frames)} frames")

    # --- Save ---
    output_path = config.get("output_clip", "exports/motion_data/retargeted_clip.json")
    if test_mode:
        output_path = "exports/motion_data/retargeted_test_poses.json"

    output = {
        "target_name": target["target_name"],
        "bone_names": tgt_names,
        "parent_indices": tgt_parents,
        "target_bind_local_pos": target["target_bind_local_pos"],
        "target_bind_local_rot": target["target_bind_local_rot"],
        "frames": frames,
    }
    save_json(output_path, output)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
