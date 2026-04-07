import os
import json
import numpy as np
import isaacgym  # keep before Isaac Gym env imports
import torch

from envs.humanoid_env import HumanoidEnv
from poselib.core.rotation3d import quat_from_angle_axis, quat_mul
from poselib.skeleton.skeleton3d import SkeletonState


# ---------------- utils ----------------

def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def normalize_quat_xyzw(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


def vec3_dict(v):
    return {
        "x": float(v[0]),
        "y": float(v[1]),
        "z": float(v[2]),
    }


def quat_dict_xyzw(q):
    q = normalize_quat_xyzw(q)
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2]),
        "w": float(q[3]),
    }


# ---------------- reference pose ----------------

def create_a_pose_bind_local_rot(skeleton, a_pose_angle_deg=45.0):
    """
    Create an A-pose reference from SkeletonState.zero_pose(skeleton).
    Returns bind_local_rot in XYZW.
    """
    ref_pose = SkeletonState.zero_pose(skeleton)
    local_rot = ref_pose.local_rotation.clone()

    left_idx = skeleton.index("left_upper_arm")
    right_idx = skeleton.index("right_upper_arm")

    print("[INFO] left_upper_arm idx:", left_idx)
    print("[INFO] right_upper_arm idx:", right_idx)

    left_rot = quat_from_angle_axis(
        angle=torch.tensor([a_pose_angle_deg], dtype=torch.float32),
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        degree=True,
    )

    right_rot = quat_from_angle_axis(
        angle=torch.tensor([-a_pose_angle_deg], dtype=torch.float32),
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        degree=True,
    )

    local_rot[left_idx] = quat_mul(left_rot, local_rot[left_idx])
    local_rot[right_idx] = quat_mul(right_rot, local_rot[right_idx])

    return to_numpy(local_rot).astype(np.float32)


def save_debug_pose(skeleton, bind_local_rot_xyzw, save_path):
    """
    Optional debug export of the constructed A-pose.
    """
    try:
        pose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=skeleton,
            r=torch.tensor(bind_local_rot_xyzw, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            is_local=True,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pose.to_file(save_path)
        print("[INFO] Saved debug pose:", save_path)
    except Exception as e:
        print("[WARNING] Failed to save debug pose:", e)


# ---------------- exporter ----------------

def main():
    # ===== config =====
    device = "cuda"
    output_path = "exports/motion_data/exported_source_motion_to_unity.json"

    a_pose_angle_deg = 45.0

    save_debug_pose_file = True
    debug_pose_path = "exports/motion_data/debug_apose.npy"
    # ==================

    print("[INFO] Creating environment...")
    env = HumanoidEnv(num_envs=1, device=device, enable_viewer=False)

    print("[INFO] Loading source motion from env.motion_lib.get_motion(0)...")
    motion = env.motion_lib.get_motion(0)
    skeleton = motion.skeleton_tree

    bone_names = list(skeleton.node_names)
    parent_indices = to_numpy(skeleton.parent_indices).astype(int)

    print("[INFO] Number of bones:", len(bone_names))
    print("[INFO] Bone names:", bone_names)

    # bind local positions from skeleton
    bind_local_pos = to_numpy(skeleton.local_translation).astype(np.float32)

    # build A-pose bind rotations
    bind_local_rot = create_a_pose_bind_local_rot(
        skeleton=skeleton,
        a_pose_angle_deg=a_pose_angle_deg,
    )

    print("[INFO] Using constructed A-pose as bind_local_rot")

    if save_debug_pose_file:
        save_debug_pose(
            skeleton=skeleton,
            bind_local_rot_xyzw=bind_local_rot,
            save_path=debug_pose_path,
        )

    # motion data from env.motion_lib
    # IMPORTANT: original/source motion quaternions are effectively WXYZ,
    # so convert them to Unity-friendly XYZW before export.
    local_rot = to_numpy(motion.local_rotation).astype(np.float32)          # [T, J, 4], WXYZ                       # [T, J, 4], XYZW

    root_pos = to_numpy(motion.global_translation[:, 0]).astype(np.float32)     # [T, 3]

    num_frames, num_bones = local_rot.shape[:2]
    fps = float(motion.fps)

    print(f"[INFO] Frames: {num_frames}, Bones: {num_bones}, FPS: {fps}")

    frames = []
    for t in range(num_frames):
        frames.append({
            "root_pos": vec3_dict(root_pos[t]),
            "local_rot": [quat_dict_xyzw(local_rot[t, j]) for j in range(num_bones)],
        })

    clip = {
        "format": "isaac_source_joint_local_unity",
        "fps": fps,
        "num_frames": num_frames,
        "num_bones": num_bones,
        "bone_names": bone_names,
        "parent_indices": parent_indices.tolist(),
        "bind_local_pos": [vec3_dict(bind_local_pos[j]) for j in range(num_bones)],
        "bind_local_rot": [quat_dict_xyzw(bind_local_rot[j]) for j in range(num_bones)],
        "frames": frames,
        "meta": {
            "quaternion_export_format": "xyzw",
            "motion_source_quaternion_input_format": "xyzw",
            "bind_pose_type": "A-pose",
            "a_pose_angle_deg": float(a_pose_angle_deg),
            "source_motion_origin": "env.motion_lib.get_motion(0)",
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clip, f, indent=2)

    print("\n[INFO] Export complete:", output_path)


if __name__ == "__main__":
    main()