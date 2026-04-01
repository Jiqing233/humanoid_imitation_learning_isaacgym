import os
import json
import numpy as np
import isaacgym  # keep this before Isaac Gym env imports

from envs.humanoid_env import HumanoidEnv


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float32)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def vec3_dict(v):
    return {
        "x": float(v[0]),
        "y": float(v[1]),
        "z": float(v[2]),
    }


def quat_dict_xyzw(q):
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2]),
        "w": float(q[3]),
    }


def main():
    device = "cuda"
    output_path = "exports/motion_data/source_motion_to_unity.json"

    env = HumanoidEnv(num_envs=1, device=device, enable_viewer=False)

    # Clean source motion clip from MotionLib, NOT PPO rollout
    motion = env.motion_lib.get_motion(0)
    skeleton = motion.skeleton_tree

    bone_names = list(skeleton.node_names)

    if hasattr(skeleton, "parent_indices"):
        parent_indices = [int(x) for x in to_numpy(skeleton.parent_indices)]
    elif hasattr(skeleton, "_parent_indices"):
        parent_indices = [int(x) for x in to_numpy(skeleton._parent_indices)]
    else:
        raise RuntimeError("Could not find parent indices on skeleton tree.")

    bind_local_pos = to_numpy(skeleton.local_translation)   # [J, 3]

    # True source motion data
    local_rot = to_numpy(motion.local_rotation)             # [T, J, 4], likely wxyz
    root_pos = to_numpy(motion.global_translation[:, 0])    # [T, 3]

     # true source bind local rotation
    if hasattr(skeleton, "local_rotation"):
        bind_local_rot = to_numpy(skeleton.local_rotation)  # [J, 4]
        print("Using skeleton.local_rotation for bind_local_rot")
    elif hasattr(skeleton, "local_orientation"):
        bind_local_rot = to_numpy(skeleton.local_orientation)
        print("Using skeleton.local_orientation for bind_local_rot")
    else:
        raise RuntimeError(
            "Could not find true source bind local rotations on skeleton. "
            "Please inspect skeleton attributes and choose the correct field."
        )

    fps = float(motion.fps)

    num_frames = int(local_rot.shape[0])
    num_bones = int(local_rot.shape[1])

    frames = []
    for t in range(num_frames):
        frame_local_rot = []
        for j in range(num_bones):
            #q_xyzw = quat_wxyz_to_xyzw(local_rot[t, j])
            q_xyzw = local_rot[t, j]
            frame_local_rot.append(quat_dict_xyzw(q_xyzw))

        frames.append({
            "root_pos": vec3_dict(root_pos[t]),
            "local_rot": frame_local_rot,
        })

    clip = {
        "format": "isaac_source_joint_local_v1",
        "fps": fps,
        "num_frames": num_frames,
        "num_bones": num_bones,
        "bone_names": bone_names,
        "parent_indices": parent_indices,
        "bind_local_pos": [vec3_dict(bind_local_pos[j]) for j in range(num_bones)],
        #"bind_local_rot": [quat_dict_xyzw(quat_wxyz_to_xyzw(bind_local_rot[j])) for j in range(num_bones)],
        "bind_local_rot": [quat_dict_xyzw(bind_local_rot[j]) for j in range(num_bones)],
        "frames": frames,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clip, f, indent=2)

    print(f"Saved source motion JSON to: {output_path}")
    print(f"Frames: {num_frames}, Bones: {num_bones}, FPS: {fps}")


if __name__ == "__main__":
    main()