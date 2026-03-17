from envs.humanoid_env import HumanoidEnv
from ppo import ActorCritic, RolloutBuffer, PPO
import torch

device = "cuda"

# Debug with a small number first
env = HumanoidEnv(num_envs=4, device=device, enable_viewer=True)
print("Environment Observation:", env.obs_dim)

model = ActorCritic(env.obs_dim, env.act_dim).to(device)
ppo = PPO(model=model, device=device)

num_steps = 256

buffer = RolloutBuffer(
    num_steps,
    env.num_envs,
    env.obs_dim,
    env.act_dim,
    device
)

obs = env.compute_observations()
print("OBS:", obs.shape)

for iteration in range(10000):
    buffer.step = 0

    for step in range(num_steps):
        with torch.no_grad():
            action, logprob = model.act(obs)
            value = model.critic(obs).squeeze(-1)

        next_obs, reward, done = env.step(action)
        buffer.add(obs, action, logprob, reward, done, value)

        if step == 0:
            print(
                f"Iter {iteration} | "
                f"action mean {action.mean().item():.4f} | "
                f"action std {action.std().item():.4f}"
            )

        obs = next_obs
        env.render()

    with torch.no_grad():
        last_value = model.critic(obs).squeeze(-1)

    before_weight = model.actor[0].weight[0, 0].item()

    buffer.compute_returns(last_value)
    ppo.update(buffer)

    after_weight = model.actor[0].weight[0, 0].item()

    print(
        f"Iteration: {iteration} | "
        f"Reward: {buffer.rewards.mean().item():.4f} | "
        f"Done frac: {buffer.dones.float().mean().item():.4f} | "
        f"Actor w[0,0] before/after: {before_weight:.6f} -> {after_weight:.6f}"
    )

    if iteration % 100 == 0:
        torch.save(model.state_dict(), f"checkpoint_{iteration}.pt")