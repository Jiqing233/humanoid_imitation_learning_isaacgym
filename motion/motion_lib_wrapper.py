import numpy as np
import torch
from .motion_lib import MotionLib


class MotionLibWrapper(MotionLib):
    def _calc_frame_blend(self, time, length, num_frames, dt):
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