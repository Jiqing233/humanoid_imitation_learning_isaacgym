from isaacgym import gymapi

# Acquire gym
gym = gymapi.acquire_gym()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Create environment
env = gym.create_env(sim,
                     gymapi.Vec3(-2, 0, -2),
                     gymapi.Vec3(2, 2, 2),
                     1)

# Create box asset
asset_options = gymapi.AssetOptions()
asset_options.density = 500.0
box_asset = gym.create_box(sim, 0.2, 0.2, 0.2, asset_options)

# Spawn box
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 5, 0)

box_actor = gym.create_actor(env, box_asset, pose, "box", 0, 1)

# Viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

cam_pos = gymapi.Vec3(1, 2, 5)
cam_target = gymapi.Vec3(0, 1, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)