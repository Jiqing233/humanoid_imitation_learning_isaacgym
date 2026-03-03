from envs.humanoid_env import HumanoidEnv
from isaacgym import gymapi
import torch

env = HumanoidEnv(1024)

viewer = env.gym.create_viewer(env.sim, gymapi.CameraProperties())

#Computing the center of the grids
grid_size = int(env.num_envs ** 0.5)
spacing = 2.0
center = (grid_size - 1) * spacing / 2.0
cam_pos = gymapi.Vec3(center + 15.0, center + 15.0, 10.0)
cam_target = gymapi.Vec3(center, center, 1.0)

env.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not env.gym.query_viewer_has_closed(viewer):

    actions = torch.zeros(env.num_envs, env.act_dim, device="cuda")

    env.step(actions)

    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(viewer, env.sim, True)