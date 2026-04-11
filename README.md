## Environment Setup

### 1. Create a Conda Environment

Create and activate a Python 3.8 conda environment:

```bash
conda create -n isaac_gym python=3.8
conda activate isaac_gym
```

### 2. Install PyTorch 2.0

Install PyTorch with CUDA 11.7 support:

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. Install Isaac Gym Preview 4
Go to the NVIDIA developer website:
[Isaac Gym Download Page](https://developer.nvidia.com/isaac-gym)

Follow and Download: IsaacGym_Preview_4_Package.tar.gz, unzip, cd to python/ directory, install by:

```bash
pip install -e .
```
Test installation is complete:

```bash
cd examples/
python joint_monkey.py
```

## Motion Conversion (Isaac Gym to Unity)

Convert Isaac Gym humanoid motion data (`.npy`) to Unity avatar animation format.

### 1. Export the Target Skeleton from Unity

In Unity, export your avatar's bind pose as a JSON file with this structure:

```json
{
  "target_name": "Your Avatar Name",
  "bone_names": ["Hips", "Spine 1", "..."],
  "parent_indices": [-1, 0, "..."],
  "target_bind_local_pos": [{"x": 0, "y": 0, "z": 0}, "..."],
  "target_bind_local_rot": [{"x": 0, "y": 0, "z": 0, "w": 1}, "..."]
}
```

- `bone_names`: list of all bone names in the avatar's humanoid rig.
- `parent_indices`: for each bone, the index of its parent (-1 for the root).
- `target_bind_local_pos`: each bone's local position relative to its parent in bind pose.
- `target_bind_local_rot`: each bone's local rotation (XYZW quaternion) in bind pose.

Save it to `unity/` (e.g. `unity/target_skeleton_main.json`). See `unity/target_skeleton_main.json` for a working example.

### 2. Create a Retarget Config

Create a JSON config in `configs/` that maps Isaac Gym bones to your avatar's bones. See `configs/isaac_to_banana.json` for a working example.

The Isaac Gym humanoid has 15 bones:

```
pelvis (root)
├── torso
│   ├── head
│   ├── right_upper_arm → right_lower_arm → right_hand
│   └── left_upper_arm  → left_lower_arm  → left_hand
├── right_thigh → right_shin → right_foot
└── left_thigh  → left_shin  → left_foot
```

The config file has these fields:

```json
{
  "source_clip": "exports/motion_data/source_clip.json",
  "target_skeleton": "unity/target_skeleton_main.json",
  "output_clip": "exports/motion_data/transferred_source_clip.json",

  "direct_map": { ... },
  "chain_map": { ... },
  "distribution_weights": { ... },
  "bone_corrections_xyzw": { ... }
}
```

**`target_skeleton`**: path to the Unity skeleton JSON you exported in step 1.

**`direct_map`**: 1-to-1 bone mappings. Map each Isaac bone to the single corresponding bone in your avatar:

```json
"direct_map": {
  "pelvis": "Hips",
  "head": "Head",
  "right_upper_arm": "Right Arm",
  "right_lower_arm": "Right Forearm",
  "right_hand": "Right Hand",
  "left_upper_arm": "Left Arm",
  "left_lower_arm": "Left Forearm",
  "left_hand": "Left Hand",
  "right_thigh": "Right Thigh",
  "right_shin": "Right Leg",
  "right_foot": "Right Foot",
  "left_thigh": "Left Thigh",
  "left_shin": "Left Leg",
  "left_foot": "Left Foot"
}
```

**`chain_map`**: 1-to-many mappings for when one Isaac bone corresponds to multiple avatar bones. The Isaac skeleton has a single `torso` bone, but most Unity avatars split this into multiple spine bones (Spine 1, Spine 2, Spine 3). The chain map distributes the source rotation across the chain:

```json
"chain_map": {
  "torso": ["Spine 1", "Spine 2", "Spine 3"]
}
```

**`distribution_weights`**: how to split the rotation for each chain. Weights are per-bone (not cumulative) and should sum to 1.0. For example, `[0.2, 0.35, 0.45]` means Spine 1 gets 20% of the torso rotation, Spine 2 gets 35%, and Spine 3 gets 45%:

```json
"distribution_weights": {
  "torso": [0.2, 0.35, 0.45]
}
```

**`bone_corrections_xyzw`**: optional per-bone correction quaternions (XYZW). Use `[0, 0, 0, 1]` (identity) for no correction. These can fine-tune individual bone orientations if needed.

**Tips for a new avatar:**
- Most humanoid avatars use similar bone names (Hips, Spine, Arm, etc.), but check your avatar's actual bone names in Unity.
- Bones in your avatar with no Isaac counterpart (e.g. Neck, Shoulder, toes) should be left unmapped — they will keep their bind rotation.
- Avoid chain-mapping the arms through Shoulder bones. The Isaac skeleton has no separate clavicle, so map `upper_arm` directly to `Arm` and leave `Shoulder` unmapped.
- If your avatar has a different number of spine bones, adjust the `chain_map` and `distribution_weights` accordingly.

### 3. Extract the Isaac Gym Static Skeleton

```bash
conda run -n isaac_gym python extract_isaac_skeleton.py
```

This parses `assets/humanoid.xml` and outputs `exports/isaac_static_skeleton.json`. Only needs to be run once.

### 4. Convert Motion

```bash
conda run -n isaac_gym python convert_motion.py data/martial_arts/Garren_Knifehand_amp.npy --config configs/isaac_to_banana.json
```

This outputs:
- `exports/motion_data/Garren_Knifehand_amp_unity.json` — Unity animation clip
- `exports/Garren_Knifehand_amp_viewer.html` — interactive side-by-side viewer for validation

### 5. Validate with Skeleton Viewer (Optional)

Compare the static bind poses of the source and target skeletons:

```bash
conda run -n isaac_gym python visualize_skeletons.py
```

Opens `exports/skeleton_viewer.html` in a browser.

