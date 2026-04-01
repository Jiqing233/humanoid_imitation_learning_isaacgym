import numpy as np
import isaacgym
from envs.humanoid_env import HumanoidEnv

def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)

def main():
    env = HumanoidEnv(num_envs=1, device="cuda", enable_viewer=False)
    motion = env.motion_lib.get_motion(0)

    zp = motion.zero_pose(motion.skeleton_tree, fps=60)
    print("type(zero_pose(...)) =", type(zp))

    print("\n=== zero_pose(...) attributes ===")
    for k in dir(zp):
        if not k.startswith("__"):
            print(k)

    print("\n=== zero_pose(...) candidate shapes ===")
    for k in dir(zp):
        if k.startswith("__"):
            continue
        try:
            value = getattr(zp, k)
            arr = to_numpy(value)
            if hasattr(arr, "shape"):
                print(f"{k}: shape={arr.shape}")
        except:
            pass

if __name__ == "__main__":
    main()