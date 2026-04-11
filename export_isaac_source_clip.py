import os
import json
import numpy as np
import isaacgym  # keep before Isaac Gym env imports
import torch

from envs.humanoid_env import HumanoidEnv
from poselib.core.rotation3d import quat_from_angle_axis, quat_mul
from poselib.skeleton.skeleton3d import SkeletonState


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


def create_a_pose_bind_local_rot(skeleton, a_pose_angle_deg=45.0):
    """
    Build a simple A-pose bind rotation from SkeletonState.zero_pose(skeleton).
    Assumes bones named:
      - left_upper_arm
      - right_upper_arm
    """
    ref_pose = SkeletonState.zero_pose(skeleton)
    local_rot = ref_pose.local_rotation.clone()

    left_idx = skeleton.index("left_upper_arm")
    right_idx = skeleton.index("right_upper_arm")

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


def main():
    # ---------------- config ----------------
    device = "cuda"
    clip_index = 0
    output_path = "exports/motion_data/source_clip.json"
    source_name = "isaac_source"
    a_pose_angle_deg = 45.0
    # ----------------------------------------

    print("[INFO] Creating environment...")
    env = HumanoidEnv(num_envs=1, device=device, enable_viewer=False)

    print(f"[INFO] Loading source motion clip {clip_index} from env.motion_lib...")
    motion = env.motion_lib.get_motion(clip_index)

    skeleton = motion.skeleton_tree
    bone_names = list(skeleton.node_names)
    parent_indices = to_numpy(skeleton.parent_indices).astype(int)
    source_bind_local_pos = to_numpy(skeleton.local_translation).astype(np.float32)
    source_bind_local_rot = create_a_pose_bind_local_rot(
        skeleton=skeleton,
        a_pose_angle_deg=a_pose_angle_deg,
    )

    source_local_rot = to_numpy(motion.local_rotation).astype(np.float32)   # [T, J, 4]
    root_pos = to_numpy(motion.global_translation[:, 0]).astype(np.float32) # [T, 3]

    num_frames, num_bones = source_local_rot.shape[:2]

    print(f"[INFO] Frames: {num_frames}")
    print(f"[INFO] Bones: {num_bones}")
    print("[INFO] Bone names:", bone_names)

    frames = []
    for t in range(num_frames):
        frame = {
            "root_pos": vec3_dict(root_pos[t]),
            "source_local_rot": [
                quat_dict_xyzw(source_local_rot[t, j]) for j in range(num_bones)
            ],
        }
        frames.append(frame)

    clip = {
        "source_name": source_name,
        "bone_names": bone_names,
        "parent_indices": parent_indices.tolist(),
        "source_bind_local_pos": [
            vec3_dict(source_bind_local_pos[j]) for j in range(num_bones)
        ],
        "source_bind_local_rot": [
            quat_dict_xyzw(source_bind_local_rot[j]) for j in range(num_bones)
        ],
        "frames": frames,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clip, f, indent=2)

    print("[INFO] Export complete:", output_path)


if __name__ == "__main__":
    main()