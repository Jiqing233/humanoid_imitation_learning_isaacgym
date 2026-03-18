import isaacgym
import numpy as np
import torch

from motion.motion_lib import MotionLib

# monkey-patch MotionLib._calc_frame_blend without modifying motion_lib.py
def patched_calc_frame_blend(self, time, length, num_frames, dt):
    if isinstance(time, torch.Tensor):
        time = time.detach().cpu().numpy()
    if isinstance(length, torch.Tensor):
        length = length.detach().cpu().numpy()
    if isinstance(num_frames, torch.Tensor):
        num_frames = num_frames.detach().cpu().numpy()
    if isinstance(dt, torch.Tensor):
        dt = dt.detach().cpu().numpy()

    time = np.asarray(time)
    length = np.asarray(length)
    num_frames = np.asarray(num_frames)
    dt = np.asarray(dt)

    phase = time / length
    phase = np.clip(phase, 0.0, 1.0)

    frame_idx0 = (phase * (num_frames - 1)).astype(int)
    frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)

    blend = (time - frame_idx0 * dt) / dt
    blend = np.clip(blend, 0.0, 1.0)

    return frame_idx0, frame_idx1, blend


MotionLib._calc_frame_blend = patched_calc_frame_blend


motion = MotionLib(
    motion_file="data/martial_arts/amp_humanoid_walk.npy",
    num_dofs=28,
    key_body_ids=torch.tensor([5, 8, 11, 14], dtype=torch.long),
    device="cuda"
)

motion_ids = torch.zeros(1, dtype=torch.long)
motion_times = torch.tensor([0.0], dtype=torch.float32)

root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = \
    motion.get_motion_state(motion_ids, motion_times)

print("root_pos shape:", root_pos.shape)
print("root_rot shape:", root_rot.shape)
print("dof_pos shape:", dof_pos.shape)
print("root_vel shape:", root_vel.shape)
print("root_ang_vel shape:", root_ang_vel.shape)
print("dof_vel shape:", dof_vel.shape)
print("key_pos shape:", key_pos.shape)
