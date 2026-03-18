from envs.humanoid_env import HumanoidEnv
import torch

env = HumanoidEnv(num_envs=1, device="cuda", enable_viewer=True)
obs = env.reset()

while True:
    actions = torch.zeros(1, env.act_dim, device="cuda")
    obs, reward, done = env.step(actions)
    env.render()