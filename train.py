from envs.humanoid_env import HumanoidEnv

env = HumanoidEnv(num_envs=1024)

for iteration in range(10000):

    obs = env.compute_observations()

    actions = torch.randn(env.num_envs, env.act_dim, device="cuda")

    obs, reward, done = env.step(actions)

    print(reward.mean())