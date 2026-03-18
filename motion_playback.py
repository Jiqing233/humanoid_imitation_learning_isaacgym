from isaacgym import gymtorch
from envs.humanoid_env import HumanoidEnv
import torch

device = "cuda"

env = HumanoidEnv(num_envs=1, device=device, enable_viewer=True)

if env.dof_pos_ref is None:
    raise RuntimeError("No dof_pos_ref found in motion file.")

print("num_dofs:", env.num_dofs)
print("dof_pos_ref shape:", env.dof_pos_ref.shape)

kp = 100.0
kd = 10.0

while True:
    for f in range(env.motion_length):
        env.motion_phase[:] = f

        # current dof state
        dof_view = env.dof_states.view(env.num_envs, env.num_dofs, 2)
        q = dof_view[:, :, 0]
        qd = dof_view[:, :, 1]

        # target from reference motion
        q_target = env.dof_pos_ref[f].unsqueeze(0)

        if env.dof_vel_ref is not None:
            qd_target = env.dof_vel_ref[f].unsqueeze(0)
        else:
            qd_target = torch.zeros_like(q_target)

        # manual PD in torque mode
        torque = kp * (q_target - q) + kd * (qd_target - qd)

        # clamp by effort limits if you want
        torque = torch.clamp(torque, -300.0, 300.0)

        env.gym.set_dof_actuation_force_tensor(
            env.sim,
            gymtorch.unwrap_tensor(torque.contiguous().view(-1))
        )

        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        env.render()