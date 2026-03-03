import os
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

class HumanoidEnv:

    def __init__(self, num_envs=1024, device="cuda"):

        self.num_envs = num_envs
        self.device = device

        self._create_sim()
        self._create_envs()
        self._prepare_tensors()

        self.obs_dim = self.root_states.shape[1] + self.dof_states.view(self.num_envs, -1).shape[1]
        self.act_dim = self.num_dofs

    def _create_sim(self):
        self.gym = gymapi.acquire_gym()
        gymutil.parse_arguments()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        asset_root = os.path.join(project_dir, "assets")
        asset_file = "humanoid.xml"
        print("Loading from:", os.path.join(asset_root, asset_file))

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.replace_cylinder_with_capsule = True

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        if humanoid_asset is None:
            raise RuntimeError("Failed to load humanoid asset. Check that Assets/humanoid.xml exists.")

        print("DOFs:", self.gym.get_asset_dof_count(humanoid_asset))
        print("Bodies:", self.gym.get_asset_rigid_body_count(humanoid_asset))

        self.num_dofs = self.gym.get_asset_dof_count(humanoid_asset)

        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1.0)

            actor = self.gym.create_actor(env, humanoid_asset, pose, "humanoid", i, 1)

            self.envs.append(env)
            self.actors.append(actor)

        self.gym.prepare_sim(self.sim)

    def _prepare_tensors(self):

        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(root_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_tensor)

    def step(self, actions):

        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            gymtorch.unwrap_tensor(actions)
        )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        obs = self.compute_observations()
        reward = self.compute_reward()
        done = self.compute_done()

        return obs, reward, done

    def compute_observations(self):

        dof_view = self.dof_states.view(self.num_envs, -1)

        obs = torch.cat([
            self.root_states,
            dof_view
        ], dim=-1)

        return obs

    def compute_reward(self):
        height = self.root_states[:, 2]
        return height

    def compute_done(self):
        return self.root_states[:, 2] < 0.5