import os
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch


class HumanoidEnv:
    def __init__(self, num_envs=1024, device="cuda", enable_viewer=False):
        self.num_envs = num_envs
        self.device = device
        self.enable_viewer = enable_viewer
        self.viewer = None

        self._create_sim()
        self._create_envs()
        self._prepare_tensors()
        self._load_motion()

        # Episode bookkeeping
        self.max_episode_length = 300
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Observation dimension
        dof_dim = self.dof_states.view(self.num_envs, -1).shape[1]
        ref_rot_dim = self.rot_ref.shape[1] * self.rot_ref.shape[2]
        ref_lin_vel_dim = self.lin_vel_ref.shape[1] * self.lin_vel_ref.shape[2]
        ref_ang_vel_dim = self.ang_vel_ref.shape[1] * self.ang_vel_ref.shape[2]

        self.obs_dim = (
            self.root_states.shape[1] +   # current root state: 13
            dof_dim +                     # current DOF state: num_dofs * 2
            1 +                           # phase
            3 +                           # target root pos
            ref_rot_dim +
            ref_lin_vel_dim +
            ref_ang_vel_dim
        )

        self.act_dim = self.num_dofs

        if self.enable_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("Failed to create viewer")

            grid_size = int(np.sqrt(self.num_envs))
            spacing = 2.0
            center = (grid_size - 1) * spacing / 2.0
            cam_pos = gymapi.Vec3(center + 15.0, center + 15.0, 10.0)
            cam_target = gymapi.Vec3(center, center, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_sim(self):
        self.gym = gymapi.acquire_gym()
        gymutil.parse_arguments()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        asset_root = os.path.join(project_dir, "assets")
        asset_file = "humanoid.xml"

        print("Loading asset from:", os.path.join(asset_root, asset_file))

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.replace_cylinder_with_capsule = True

        humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        if humanoid_asset is None:
            raise RuntimeError("Failed to load humanoid asset")

        self.num_dofs = self.gym.get_asset_dof_count(humanoid_asset)
        num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)

        print("DOFs:", self.num_dofs)
        print("Bodies:", num_bodies)

        # Explicitly configure DOF properties
        dof_props = self.gym.get_asset_dof_properties(humanoid_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(0.0)
        dof_props["effort"].fill(300.0)

        print("DOF effort limits:", dof_props["effort"][:10])

        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actors = []

        grid_size = int(np.sqrt(self.num_envs))

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, grid_size)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)

            actor = self.gym.create_actor(env, humanoid_asset, pose, "humanoid", i, 1)
            self.gym.set_actor_dof_properties(env, actor, dof_props)

            self.envs.append(env)
            self.actors.append(actor)

        self.gym.prepare_sim(self.sim)

    def _prepare_tensors(self):
        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.root_states = gymtorch.wrap_tensor(root_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_tensor)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Save stable initial simulator state
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

    def _load_motion(self):
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        motion_path = os.path.join(
            project_dir, "data", "martial_arts", "amp_humanoid_walk.npy"
        )

        motion_data = np.load(motion_path, allow_pickle=True)
        motion_dict = motion_data.item()

        root_pos = motion_dict["root_translation"]["arr"]
        rotation = motion_dict["rotation"]["arr"]
        lin_vel = motion_dict["global_velocity"]["arr"]
        ang_vel = motion_dict["global_angular_velocity"]["arr"]

        self.root_pos_ref = torch.tensor(root_pos, dtype=torch.float32, device=self.device)
        self.rot_ref = torch.tensor(rotation, dtype=torch.float32, device=self.device)
        self.lin_vel_ref = torch.tensor(lin_vel, dtype=torch.float32, device=self.device)
        self.ang_vel_ref = torch.tensor(ang_vel, dtype=torch.float32, device=self.device)

        self.motion_length = self.root_pos_ref.shape[0]

        # Start all envs from phase 0
        self.motion_phase = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Align reference root trajectory to each env spawn
        self.root_pos_offset = (
            self.initial_root_states[:, 0:3] - self.root_pos_ref[0].unsqueeze(0)
        )

        # Optional DOF refs if present
        self.dof_pos_ref = None
        self.dof_vel_ref = None

        possible_dof_pos_keys = ["dof_pos", "joint_pose", "pose", "local_rotation"]
        possible_dof_vel_keys = ["dof_vel", "joint_velocity", "velocity", "local_velocity"]

        for k in possible_dof_pos_keys:
            if k in motion_dict:
                self.dof_pos_ref = torch.tensor(
                    motion_dict[k]["arr"], dtype=torch.float32, device=self.device
                )
                print("Loaded DOF pose key:", k, self.dof_pos_ref.shape)
                break

        for k in possible_dof_vel_keys:
            if k in motion_dict:
                self.dof_vel_ref = torch.tensor(
                    motion_dict[k]["arr"], dtype=torch.float32, device=self.device
                )
                print("Loaded DOF vel key:", k, self.dof_vel_ref.shape)
                break

        print("Motion frames:", self.motion_length)
        print("root_pos_ref:", self.root_pos_ref.shape)
        print("rot_ref:", self.rot_ref.shape)
        print("lin_vel_ref:", self.lin_vel_ref.shape)
        print("ang_vel_ref:", self.ang_vel_ref.shape)

        if self.dof_pos_ref is None:
            print("WARNING: DOF pose reference not found in motion file.")
        if self.dof_vel_ref is None:
            print("WARNING: DOF velocity reference not found in motion file.")

    def render(self):
        if self.viewer is None:
            return

        if self.gym.query_viewer_has_closed(self.viewer):
            raise SystemExit("Viewer closed")

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return self.compute_observations()

        # Sample an initialization frame from the first few clip frames
        max_init_frame = min(10, self.motion_length)
        frame_ids = torch.randint(
            0, max_init_frame, (env_ids.numel(),),
            device=self.device, dtype=torch.long
        )

        self.motion_phase[env_ids] = frame_ids
        self.progress_buf[env_ids] = 0

        # Reference root state
        ref_root_pos = self.root_pos_ref[frame_ids]               # [M, 3]
        ref_root_rot = self.rot_ref[frame_ids, 0, :]             # [M, 4]
        ref_root_lin_vel = self.lin_vel_ref[frame_ids, 0, :]     # [M, 3]
        ref_root_ang_vel = self.ang_vel_ref[frame_ids, 0, :]     # [M, 3]

        aligned_root_pos = ref_root_pos + self.root_pos_offset[env_ids]

        self.root_states[env_ids, 0:3] = aligned_root_pos
        self.root_states[env_ids, 3:7] = ref_root_rot
        self.root_states[env_ids, 7:10] = ref_root_lin_vel
        self.root_states[env_ids, 10:13] = ref_root_ang_vel

        # DOF state
        dof_view = self.dof_states.view(self.num_envs, self.num_dofs, 2)

        if self.dof_pos_ref is not None and self.dof_vel_ref is not None:
            dof_view[env_ids, :, 0] = self.dof_pos_ref[frame_ids]
            dof_view[env_ids, :, 1] = self.dof_vel_ref[frame_ids]
        else:
            init_dof_view = self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)
            dof_view[env_ids] = init_dof_view[env_ids]

        actor_indices = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            actor_indices.numel()
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_indices),
            actor_indices.numel()
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return self.compute_observations()

    def step(self, actions):
        self.progress_buf += 1

        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        torque = (5.0 * clipped_actions).contiguous().view(-1)

        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torque)
        )

        self.motion_phase += 1
        self.motion_phase %= self.motion_length

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        reward = self.compute_reward()
        done = self.compute_done()

        done_env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            obs = self.reset(done_env_ids)
        else:
            obs = self.compute_observations()

        return obs, reward, done

    def compute_observations(self):
        dof_view = self.dof_states.view(self.num_envs, -1)

        phase = self.motion_phase.float() / self.motion_length
        phase = phase.unsqueeze(-1)

        ref_root_pos = self.root_pos_ref[self.motion_phase]  # [N, 3]
        ref_rot = self.rot_ref[self.motion_phase].reshape(self.num_envs, -1)
        ref_lin_vel = self.lin_vel_ref[self.motion_phase].reshape(self.num_envs, -1)
        ref_ang_vel = self.ang_vel_ref[self.motion_phase].reshape(self.num_envs, -1)

        obs = torch.cat([
            self.root_states,   # current root state
            dof_view,           # current dof state
            phase,              # current motion phase
            ref_root_pos,       # target root position
            ref_rot,            # target body rotations
            ref_lin_vel,        # target body linear velocities
            ref_ang_vel         # target body angular velocities
        ], dim=-1)

        return obs

    def compute_reward(self):
        rb = self.rb_states.view(self.num_envs, 15, 13)

        body_rot = rb[:, :, 3:7]
        body_lin_vel = rb[:, :, 7:10]

        ref_rot = self.rot_ref[self.motion_phase]
        ref_lin_vel = self.lin_vel_ref[self.motion_phase]

        # Rotation similarity
        rot_dot = torch.sum(body_rot * ref_rot, dim=-1).abs()
        rot_reward = rot_dot.mean(dim=1)

        # Velocity tracking
        lin_vel_error = torch.mean(
            torch.sum((body_lin_vel - ref_lin_vel) ** 2, dim=-1), dim=1
        )
        lin_vel_reward = torch.exp(-0.1 * lin_vel_error)

        # Alive/upright reward
        alive_reward = (self.root_states[:, 2] > 0.75).float()

        reward = (
            0.75 * rot_reward +
            0.15 * lin_vel_reward +
            0.10 * alive_reward
        )

        return reward

    def compute_done(self):
        fell = self.root_states[:, 2] < 0.5
        timeout = self.progress_buf >= self.max_episode_length
        return fell | timeout