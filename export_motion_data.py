import os
import numpy as np
import isaacgym
import torch
import json

from envs.humanoid_env import HumanoidEnv
from ppo import ActorCritic


def main():
    device = "cuda"
    checkpoint_path = "exports/checkpoints/Garren_Knifehand/checkpoint_600.pt"   # change this
    output_path = "exports/motion_data/exported_motion.npz"

    # Use a single env for clean export
    env = HumanoidEnv(num_envs=1, device=device, enable_viewer=True)
    body_parent = [
        -1, 0, 1,
        1, 3, 4,
        1, 6, 7,
        0, 9, 10,
        0, 12, 13
    ]

    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Reset env
    obs = env.reset()

    # Decide how many frames to export
    num_frames = 200

    # Containers
    root_states_seq = []
    dof_pos_seq = []
    dof_vel_seq = []
    rb_pos_seq = []
    rb_rot_seq = []
    rb_lin_vel_seq = []
    rb_ang_vel_seq = []
    pd_target_seq = []
    phase_seq = []
    time_seq = []

    with torch.no_grad():
        for f in range(num_frames):
            action, _ = model.act(obs)
            obs, reward, done = env.step(action)

            # root state: [1, 13]
            root_state = env.root_states[0].detach().cpu().numpy().copy()

            # dof state: [num_envs * num_dofs, 2] -> [1, num_dofs, 2]
            dof_view = env.dof_states.view(env.num_envs, env.num_dofs, 2)
            dof_pos = dof_view[0, :, 0].detach().cpu().numpy().copy()
            dof_vel = dof_view[0, :, 1].detach().cpu().numpy().copy()

            # rigid body state: [num_envs, num_bodies, 13]
            rb = env.rb_states.view(env.num_envs, -1, 13)
            rb_pos = rb[0, :, 0:3].detach().cpu().numpy().copy()
            rb_rot = rb[0, :, 3:7].detach().cpu().numpy().copy()
            rb_lin_vel = rb[0, :, 7:10].detach().cpu().numpy().copy()
            rb_ang_vel = rb[0, :, 10:13].detach().cpu().numpy().copy()

            pd_target = env.pd_targets[0].detach().cpu().numpy().copy()
            phase = int(env.motion_phase[0].item())
            motion_time = float(env.motion_times[0].item())

            root_states_seq.append(root_state)
            dof_pos_seq.append(dof_pos)
            dof_vel_seq.append(dof_vel)
            rb_pos_seq.append(rb_pos)
            rb_rot_seq.append(rb_rot)
            rb_lin_vel_seq.append(rb_lin_vel)
            rb_ang_vel_seq.append(rb_ang_vel)
            pd_target_seq.append(pd_target)
            phase_seq.append(phase)
            time_seq.append(motion_time)

            env.render()

            if done[0]:
                print(f"Episode ended at frame {f}, resetting.")
                obs = env.reset()

    np.savez(
        output_path,
        root_states=np.asarray(root_states_seq, dtype=np.float32),   # [T, 13]
        dof_pos=np.asarray(dof_pos_seq, dtype=np.float32),           # [T, num_dofs]
        dof_vel=np.asarray(dof_vel_seq, dtype=np.float32),           # [T, num_dofs]
        rb_pos=np.asarray(rb_pos_seq, dtype=np.float32),             # [T, num_bodies, 3]
        rb_rot=np.asarray(rb_rot_seq, dtype=np.float32),             # [T, num_bodies, 4]
        rb_lin_vel=np.asarray(rb_lin_vel_seq, dtype=np.float32),     # [T, num_bodies, 3]
        rb_ang_vel=np.asarray(rb_ang_vel_seq, dtype=np.float32),     # [T, num_bodies, 3]
        pd_targets=np.asarray(pd_target_seq, dtype=np.float32),      # [T, num_dofs]
        motion_phase=np.asarray(phase_seq, dtype=np.int32),          # [T]
        motion_time=np.asarray(time_seq, dtype=np.float32),          # [T]
    )

    meta = {
    "fps": 1.0 / env.gym.get_sim_params(env.sim).dt,
    "num_dofs": env.num_dofs,
    "num_bodies": int(env.rb_states.view(env.num_envs, -1, 13).shape[1]),
    "body_names": list(env.body_names),
    "dof_names": list(env.dof_names),
    "body_parent": body_parent,
    }
    print("body_names:", env.body_names)

    meta_path = output_path.replace(".npz", "_meta.json")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved metadata to: {meta_path}")

    print(f"Saved exported motion to: {output_path}")


if __name__ == "__main__":
    main()