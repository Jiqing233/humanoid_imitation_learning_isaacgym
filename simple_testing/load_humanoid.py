import numpy as np
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

# -------------------------------------------------
# 1. Initialize gym
# -------------------------------------------------
gym = gymapi.acquire_gym()

args = gymutil.parse_arguments()

# -------------------------------------------------
# 2. Create simulator
# -------------------------------------------------
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.use_gpu_pipeline = True

sim = gym.create_sim(args.compute_device_id,
                     args.graphics_device_id,
                     gymapi.SIM_PHYSX,
                     sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# -------------------------------------------------
# 3. Add ground
# -------------------------------------------------
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# -------------------------------------------------
# 4. Load humanoid (URDF; more reliable than MJCF in Isaac Gym)
# -------------------------------------------------
project_dir = os.path.dirname(os.path.abspath(__file__))
asset_root = os.path.join(project_dir, "assets")
asset_file = "humanoid.xml"
print("Loading from:", os.path.join(asset_root, asset_file))

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False   # IMPORTANT
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
asset_options.replace_cylinder_with_capsule = True

humanoid_asset = gym.load_asset(sim,
                                asset_root,
                                asset_file,
                                asset_options)

if humanoid_asset is None:
    raise RuntimeError("Failed to load humanoid asset. Check that Assets/humanoid.xml exists.")

print("DOFs:", gym.get_asset_dof_count(humanoid_asset))
print("Bodies:", gym.get_asset_rigid_body_count(humanoid_asset))

# -------------------------------------------------
# 5. Create environment
# -------------------------------------------------
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

env = gym.create_env(sim, env_lower, env_upper, 1)

# -------------------------------------------------
# 6. Spawn humanoid
# -------------------------------------------------
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1.0)  # start slightly above ground

actor_handle = gym.create_actor(env,
                                humanoid_asset,
                                pose,
                                "humanoid",
                                0,
                                1)
gym.prepare_sim(sim)
# -------------------------------------------------
# 7. Viewer
# -------------------------------------------------
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# -------------------------------------------------
# 8. Simulation loop
# -------------------------------------------------
while not gym.query_viewer_has_closed(viewer):

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)